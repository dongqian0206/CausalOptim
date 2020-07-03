import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from causal_meta.utils.data_utils import RandomSplineSCM
from causal_meta.modules.encoder import Rotor


def sample_from_normal(mean, std, nsamples, n_features):
    return torch.normal(torch.ones(nsamples, n_features) * mean,
                        torch.ones(nsamples, n_features) * std)


def init_configs():
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--mle_nsamples', type=int, default=1000)
    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1111)
    args = parser.parse_args()
    return args


def main():
    args = init_configs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    scm = RandomSplineSCM(span=8,
                          num_anchors=10,
                          order=2,
                          range_scale=1.,
                          input_noise=False, output_noise=True)

    with torch.no_grad():

        plt.figure(figsize=(9, 5))
        ax = plt.subplot(1, 1, 1)
        mus = [0, -4., 4.]
        colors = ['C0', 'C3', 'C2']
        labels = [r'Training ($\mu = 0$)', r'Transfer ($\mu = -4$)', r'Transfer ($\mu = +4$)']

        for i, (mu, color, label) in enumerate(zip(mus, colors, labels)):
            x = sample_from_normal(mu, 2, args.mle_nsamples, args.n_features)
            y = scm(x)
            ax.scatter(x.squeeze(1).numpy(), y.squeeze(1).numpy(),
                       color=color, marker='+', alpha=0.5, label=label, zorder=2 - i)

        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.legend(loc=1, prop={'size': 13})
        ax.set_xlabel('A', fontsize=14)
        ax.set_ylabel('B', fontsize=14)
        plt.show()

    encoder = Rotor(0.758)
    decoder = Rotor(-0.5 * np.pi / 2)

    with torch.no_grad():
        x = sample_from_normal(0, 2, args.mle_nsamples, args.n_features)
        y = scm(x)

        x_dec, y_dec = decoder(x, y)

        x_enc, y_enc = encoder(x_dec, y_dec)

        plt.figure(figsize=(9, 9))
        ax = plt.subplot(3, 1, 1)
        ax.scatter(x.squeeze(1).numpy(), y.squeeze(1).numpy(),
                   color='C0', marker='+', alpha=0.5, label=r'$A \rightarrow B$')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(loc=3, prop={'size': 13})

        ax = plt.subplot(3, 1, 2)
        ax.scatter(x_dec.squeeze(1).numpy(), y_dec.squeeze(1).numpy(),
                   color='C2', marker='+', alpha=0.5, label=r'$X \rightarrow Y$')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(loc=3, prop={'size': 13})

        ax = plt.subplot(3, 1, 3)
        ax.scatter(x_enc.squeeze(1).numpy(), y_enc.squeeze(1).numpy(),
                   color='C0', marker='+', alpha=0.5, label=r'$X \rightarrow Y, \theta=0.758$')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(loc=3, prop={'size': 13})

        plt.show()


if __name__ == '__main__':
    main()
