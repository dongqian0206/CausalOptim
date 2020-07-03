import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from causal_meta.utils.data_utils import RandomSplineSCM
from causal_meta.utils.train_utils import train_mle_nll, train_alpha
from causal_meta.modules.mdn import MDN


def init_configs():
    parser = argparse.ArgumentParser(description='Causal graph')

    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--mdn_n_gaussians', type=int, default=10)
    parser.add_argument('--gmm_n_gaussians', type=int, default=10)

    parser.add_argument('--mle_n_iters', type=int, default=3000)
    parser.add_argument('--em_n_iters', type=int, default=500)
    parser.add_argument('--mle_nsamples', type=int, default=3000)
    parser.add_argument('--mle_lr', type=float, default=1e-3)

    parser.add_argument('--meta_n_iters', type=int, default=500)
    parser.add_argument('--meta_nsamples', type=int, default=1000)
    parser.add_argument('--train_gmm', type=str, default=True)
    parser.add_argument('--alpha_lr', type=float, default=0.1)
    
    parser.add_argument('--finetune_n_iters', type=int, default=200)
    parser.add_argument('--finetune_lr', type=float, default=1e-3)

    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()

    return args


def main():

    args = init_configs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    scm = RandomSplineSCM(span=8,
                          num_anchors=10,
                          order=2,
                          range_scale=1.,
                          input_noise=False, output_noise=True)

    model_x2y = MDN(input_size=args.n_features,
                    hidden_size=args.hidden_size,
                    n_gaussians=args.mdn_n_gaussians).cuda()

    model_y2x = MDN(input_size=args.n_features,
                    hidden_size=args.hidden_size,
                    n_gaussians=args.mdn_n_gaussians).cuda()
    
    print('Start pre-training model_x2y.')
    nll_x2y = train_mle_nll(args, model_x2y, scm, polarity='X2Y')

    print('Start pre-training model_y2x.')
    nll_y2x = train_mle_nll(args, model_y2x, scm, polarity='Y2X')

    ll_x2y = -np.array(nll_x2y)
    ll_y2x = -np.array(nll_y2x)

    fig, ax = plt.subplots()
    ax.plot(ll_x2y, color='C0', label=r'$A \rightarrow B$', linewidth=2)
    ax.plot(ll_y2x, color='C3', label=r'$B \rightarrow A$', linewidth=2)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('Number of Iterations', fontsize=14)
    ax.set_ylabel(r'$\log P(D \mid \cdot \rightarrow \cdot)$', fontsize=14)
    ax.set_xlim([1, args.mle_n_iters])
    ax.legend(loc=4, prop={'size': 13})

    plt.savefig('./plot/mle.png', bbox_inches='tight', format='png')
    plt.show()

    alpha = nn.Parameter(torch.tensor(0.).cuda(), requires_grad=True)
    results = train_alpha(args, model_x2y, model_y2x, scm, alpha, mode='logsigp')

    sig_alphas = np.asarray([result.sig_alpha for result in results])

    fig, ax = plt.subplots()
    ax.plot(sig_alphas, linewidth=2, color='k', 
            label='Iters = {0}, N={0}'.format(args.finetune_n_iters, args.meta_nsamples))

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.axhline(1, c='gray', ls='--')
    ax.axhline(0, c='gray', ls='--')

    ax.set_xlabel('Number of Episodes', fontsize=14)
    ax.set_ylabel(r'$\sigma(\gamma)$', fontsize=14)
    ax.set_xlim([1, args.meta_n_iters])
    ax.legend(loc=4, prop={'size': 13})

    plt.savefig('./plot/transfer.png', bbox_inches='tight', format='png')
    plt.show()


if __name__ == '__main__':
    main()
