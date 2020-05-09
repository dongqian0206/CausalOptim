import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
from copy import deepcopy
from causal_meta.modules.gmm import GaussianMixture
from causal_meta.modules.mdn import mdn_nll


def logsumexp(a, b):
    min_, max_ = torch.min(a, b), torch.max(a, b)
    return max_ + F.softplus(min_ - max_)


def sample_from_normal(mean, std, nsamples, n_features):
    return torch.normal(torch.ones(nsamples, n_features) * mean,
                        torch.ones(nsamples, n_features) * std)


def gradnan_filter(model): 
    nan_found = False
    for param in model.parameters(): 
        nan_mask = torch.isnan(param.grad.data)
        nan_found = bool(nan_mask.any().item())
        param.grad.data[nan_mask] = 0.
    return nan_found


def train_mle_nll(args, model, scm, polarity='X2Y'):

    optimizer_mle = optim.Adam(model.parameters(), lr=args.mle_lr)

    losses = []

    for iter_num in range(1, args.mle_n_iters + 1):

        with torch.no_grad():
            x = sample_from_normal(0, 2, args.mle_nsamples, args.n_features)
            y = scm(x)

            if polarity == 'X2Y':
                inputs, targets = x, y
            elif polarity == 'Y2X':
                inputs, targets = y, x
            else:
                raise ValueError('%s does not match any known polarity.' % polarity)

            inputs, targets = inputs.cuda(), targets.cuda()

        loss_conditional = mdn_nll(model(inputs), targets)

        optimizer_mle.zero_grad()
        loss_conditional.backward()
        optimizer_mle.step()

        losses.append(loss_conditional.item())

    return losses


def marginal_nll(args, inputs):
    gmm = GaussianMixture(args.gmm_n_gaussians, args.n_features).cuda()
    gmm.fit(inputs)
    with torch.no_grad():
        loss_marginal = mdn_nll(gmm(inputs), inputs)
    return loss_marginal


def transfer_finetune(args, model_x2y, model_y2x, inputs, targets):
    
    optim_x2y = optim.Adam(model_x2y.parameters(), lr=args.finetune_lr)
    optim_y2x = optim.Adam(model_y2x.parameters(), lr=args.finetune_lr)

    loss_marg_x2y = marginal_nll(args, inputs).item() if args.train_gmm else 0.
    loss_marg_y2x = marginal_nll(args, targets).item() if args.train_gmm else 0.

    is_nan = False
    loss_x2y, loss_y2x = [], []

    for _ in range(args.finetune_n_iters):
        
        loss_cond_x2y = mdn_nll(model_x2y(inputs), targets)
        loss_cond_y2x = mdn_nll(model_y2x(targets), inputs)
        
        if torch.isnan(loss_cond_x2y).item() or torch.isnan(loss_cond_y2x).item():
            is_nan = True
            break
        
        optim_x2y.zero_grad()
        optim_y2x.zero_grad()
        
        loss_cond_x2y.backward() 
        loss_cond_y2x.backward()

        nan_in_x2y = gradnan_filter(model_x2y)
        nan_in_y2x = gradnan_filter(model_y2x)
        
        if nan_in_x2y or nan_in_y2x: 
            is_nan = True
            break
        
        optim_x2y.step()
        optim_y2x.step()

        loss_x2y.append(loss_cond_x2y.item() + loss_marg_x2y)
        loss_y2x.append(loss_cond_y2x.item() + loss_marg_y2x)

    return loss_x2y, loss_y2x, is_nan


def train_alpha(args, model_x2y, model_y2x, scm, alpha, mode='logmix'):

    optimizer_alpha = optim.Adam([alpha], lr=args.alpha_lr)

    results = []

    for meta_iter_num in range(1, args.meta_n_iters + 1):

        with torch.no_grad():
            # same mechanism (conditional distribution)
            param = np.random.uniform(-4, 4)
            x_ts = sample_from_normal(param, 2, args.meta_nsamples, args.n_features)
            y_ts = scm(x_ts)
            x_ts, y_ts = x_ts.cuda(), y_ts.cuda()

        state_x2y = deepcopy(model_x2y.state_dict())
        state_y2x = deepcopy(model_y2x.state_dict())

        # Evaluate performance
        loss_x2y, loss_y2x, return_is_nan = transfer_finetune(args, model_x2y, model_y2x, x_ts, y_ts)

        if not return_is_nan:
            loss_x2y = sum(loss_x2y) / len(loss_x2y)
            loss_y2x = sum(loss_y2x) / len(loss_y2x)

            # Estimate gradient
            if mode == 'mix':
                # sigmoid(alpha) * (-log_likelihood_1) + (1 - sigmoid(alpha)) * (-log_likelihood_2)
                loss_alpha = torch.sigmoid(alpha) * loss_x2y + (1 - torch.sigmoid(alpha)) * loss_y2x

            else:
                # log \sum_{i=1}^{2} exp[ log(sigma_{i}) + log_likelihood_{i}) ]
                log_alpha, log_1_m_alpha = F.logsigmoid(alpha), F.logsigmoid(-alpha)
                as_lse = logsumexp(log_alpha + loss_x2y, log_1_m_alpha + loss_y2x)

                if mode == 'logsigp':
                    loss_alpha = as_lse
                elif mode == 'sigp':
                    loss_alpha = as_lse.exp()
                else:
                    raise ValueError('%s does not match any known mode.' % mode)

            optimizer_alpha.zero_grad()
            loss_alpha.backward()
            
            if torch.isnan(alpha.grad.data).any(): 
                alpha.grad.data.zero_()
            
            optimizer_alpha.step()

            model_x2y.load_state_dict(state_x2y)
            model_y2x.load_state_dict(state_y2x)

            print('| Iteration: %d | Prob: %.3f | X2Y_Loss: %.3f | Y2X_Loss: %.3f'
                  % (meta_iter_num, torch.sigmoid(alpha).item(), loss_x2y, loss_y2x))

            with torch.no_grad():
                results.append(Namespace(iter_num=meta_iter_num,
                                         sig_alpha=torch.sigmoid(alpha).item(),
                                         loss_x2y=loss_x2y,
                                         loss_y2x=loss_y2x,
                                         loss_alpha=loss_alpha.item()))

        else:

            model_x2y.load_state_dict(state_x2y)
            model_y2x.load_state_dict(state_y2x)

            with torch.no_grad():
                results.append(Namespace(iter_num=meta_iter_num,
                                         sig_alpha=float('nan'),
                                         loss_x2y=float('nan'),
                                         loss_y2x=float('nan'),
                                         loss_alpha=float('nan')))

    return results
