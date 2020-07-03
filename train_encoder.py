import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import random
from argparse import Namespace
from copy import deepcopy
from causal_meta.utils.data_utils import RandomSplineSCM, sample_from_normal
from causal_meta.utils.train_utils import train_mle_nll, marginal_nll
from causal_meta.utils.plot_utils import plot_theta, plot_key
from causal_meta.utils.utils import gradnan_filter, logsumexp
from causal_meta.modules.mdn import MDN, mdn_nll
from causal_meta.modules.encoder import Rotor


def init_configs():
    parser = argparse.ArgumentParser(description='Causal graph')

    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--mdn_n_gaussians', type=int, default=10)
    parser.add_argument('--gmm_n_gaussians', type=int, default=10)

    parser.add_argument('--mle_n_iters', type=int, default=20)
    parser.add_argument('--em_n_iters', type=int, default=500)
    parser.add_argument('--mle_nsamples', type=int, default=1000)
    parser.add_argument('--mle_lr', type=float, default=0.01)

    parser.add_argument('--encoder_lr', type=float, default=0.01)

    parser.add_argument('--meta_n_iters', type=int, default=1000)
    parser.add_argument('--meta_nsamples', type=int, default=1000)
    parser.add_argument('--train_gmm', type=str, default=True)
    parser.add_argument('--alpha_lr', type=float, default=0.001)

    parser.add_argument('--finetune_n_iters', type=int, default=5)
    parser.add_argument('--finetune_lr', type=float, default=0.001)

    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()

    return args


def train_encoder(args, model_x2y, model_y2x, scm, encoder, decoder):
    alpha = nn.Parameter(torch.tensor(0.).cuda(), requires_grad=True)

    optim_encoder = optim.Adam(encoder.parameters(), args.encoder_lr)
    optim_alpha = optim.Adam([alpha], lr=args.alpha_lr)

    # print('Start pre-training model_x2y.')
    _ = train_mle_nll(args, model_x2y, scm, encoder, decoder, polarity='X2Y')

    # print('Start pre-training model_y2x.')
    _ = train_mle_nll(args, model_y2x, scm, encoder, decoder, polarity='Y2X')

    results = []

    for meta_iter_num in range(1, args.meta_n_iters + 1):

        _ = train_mle_nll(args, model_x2y, scm, encoder, decoder, polarity='X2Y')
        _ = train_mle_nll(args, model_y2x, scm, encoder, decoder, polarity='Y2X')

        # same mechanism (conditional distribution)
        param = np.random.uniform(-4, 4)
        a_ts = sample_from_normal(param, 2, args.meta_nsamples, args.n_features)
        b_ts = scm(a_ts)
        a_ts, b_ts = a_ts.cuda(), b_ts.cuda()
        with torch.no_grad():
            x_ts, y_ts = decoder(a_ts, b_ts)

        x_ts, y_ts = encoder(x_ts, y_ts)

        loss_marg_x2y = marginal_nll(args, x_ts) if args.train_gmm else 0.
        loss_marg_y2x = marginal_nll(args, y_ts) if args.train_gmm else 0.

        state_x2y = deepcopy(model_x2y.state_dict())
        state_y2x = deepcopy(model_y2x.state_dict())

        # Inner loop
        optim_x2y = optim.Adam(model_x2y.parameters(), lr=args.finetune_lr)
        optim_y2x = optim.Adam(model_y2x.parameters(), lr=args.finetune_lr)

        loss_x2y, loss_y2x = [], []
        is_nan = False

        for _ in range(args.finetune_n_iters):

            loss_cond_x2y = mdn_nll(model_x2y(x_ts), y_ts)
            loss_cond_y2x = mdn_nll(model_y2x(y_ts), x_ts)

            if torch.isnan(loss_cond_x2y).item() or torch.isnan(loss_cond_y2x).item():
                is_nan = True
                break

            optim_x2y.zero_grad()
            optim_y2x.zero_grad()

            loss_cond_x2y.backward(retain_graph=True)
            loss_cond_y2x.backward(retain_graph=True)

            nan_in_x2y = gradnan_filter(model_x2y)
            nan_in_y2x = gradnan_filter(model_y2x)

            if nan_in_x2y or nan_in_y2x:
                is_nan = True
                break

            optim_x2y.step()
            optim_y2x.step()

            loss_x2y.append(loss_cond_x2y + loss_marg_x2y)
            loss_y2x.append(loss_cond_y2x + loss_marg_y2x)

        if not is_nan:
            loss_x2y = torch.stack(loss_x2y).mean()
            loss_y2x = torch.stack(loss_y2x).mean()

            log_alpha, log_1_m_alpha = F.logsigmoid(alpha), F.logsigmoid(-alpha)
            loss = logsumexp(log_alpha + loss_x2y, log_1_m_alpha + loss_y2x)

            optim_encoder.zero_grad()
            optim_alpha.zero_grad()
            
            loss.backward()

            if torch.isnan(encoder.theta.grad.data).any():
                encoder.theta.grad.data.zero_()
            
            if torch.isnan(alpha.grad.data).any():
                alpha.grad.data.zero_()

            optim_encoder.step()
            optim_alpha.step()

            model_x2y.load_state_dict(state_x2y)
            model_y2x.load_state_dict(state_y2x)

            print('| Iteration: %d | Prob: %.3f | X2Y_Loss: %.3f | Y2X_Loss: %.3f | Theta: %.3f'
                  % (meta_iter_num, torch.sigmoid(alpha).item(), loss_x2y, loss_y2x, encoder.theta.item()))

            with torch.no_grad():
                results.append(Namespace(iter_num=meta_iter_num,
                                         sig_alpha=torch.sigmoid(alpha).item(),
                                         loss_x2y=loss_x2y,
                                         loss_y2x=loss_y2x,
                                         theta=encoder.theta.item()))

        else:

            model_x2y.load_state_dict(state_x2y)
            model_y2x.load_state_dict(state_y2x)

            with torch.no_grad():
                results.append(Namespace(iter_num=meta_iter_num,
                                         sig_alpha=float('nan'),
                                         loss_x2y=float('nan'),
                                         loss_y2x=float('nan'),
                                         theta=float('nan')))

    return results


def main():

    args = init_configs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    scm = RandomSplineSCM(span=8,
                          num_anchors=8,
                          order=3,
                          range_scale=1.,
                          input_noise=False, output_noise=True)

    model_x2y = MDN(input_size=args.n_features,
                    hidden_size=args.hidden_size,
                    n_gaussians=args.mdn_n_gaussians)

    model_y2x = MDN(input_size=args.n_features,
                    hidden_size=args.hidden_size,
                    n_gaussians=args.mdn_n_gaussians)

    decoder = Rotor(-0.5 * np.pi / 2)
    encoder = Rotor(0. * np.pi / 2)

    frames = train_encoder(args, model_x2y, model_y2x, scm, encoder, decoder)

    plot_theta(frames, decoder.theta)

    plot_key(frames, 'sig_alpha', name=r'$\sigma(\gamma)$')


if __name__ == '__main__':
    main()
