import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from causal_meta.utils.data_utils import RandomSplineSCM
from causal_meta.modules.mdn import MDN, mdn_nll
from causal_meta.utils.train_utils import train_alpha


def init_configs():
    parser = argparse.ArgumentParser(description='Causal graph')

    parser.add_argument('--capacity', type=int, default=32)
    parser.add_argument('--num_components', type=int, default=10)
    parser.add_argument('--gmm_num_components', type=int, default=10)

    parser.add_argument('--nsamples', type=int, default=2000)

    parser.add_argument('--alpha_num_iters', type=int, default=200)
    parser.add_argument('--alpha_lr', type=float, default=0.1)
    parser.add_argument('--finetune_num_iters', type=int, default=100)
    parser.add_argument('--finetune_lr', type=float, default=0.001)

    parser.add_argument('--seed', type=float, default=666)

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
                          order=3,
                          range_scale=1.,
                          input_noise=False, output_noise=True)

    model_x2y = MDN(args.capacity, args.num_components).cuda()
    model_y2x = MDN(args.capacity, args.num_components).cuda()

    alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)
    alpha_frames = train_alpha(args,
                               model_x2y, model_y2x,
                               scm,
                               alpha,
                               mdn_nll,
                               mode='logsigp')

    sig_alphas = np.array([1 - frame.sig_alpha for frame in alpha_frames])

    fig, ax = plt.subplots()
    ax.plot(sig_alphas, linewidth=2, color='k', label='N = {0}'.format(args.finetune_num_iters))

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.axhline(1, c='gray', ls='--')
    ax.axhline(0, c='gray', ls='--')

    ax.set_xlim([0, args.alpha_num_iters - 1])
    ax.set_xlabel('Number of Episodes', fontsize=14)
    ax.set_ylabel(r'$\sigma(\gamma)$', fontsize=14)
    ax.legend(loc=4, prop={'size': 13})

    plt.show()


if __name__ == '__main__':
    main()
