import matplotlib.pyplot as plt
import numpy as np


def plot_theta(frames, gt_theta):
    its, vals = zip(*[(frame.iter_num, frame.theta / (np.pi / 2)) for frame in frames])
    gt_theta = -gt_theta.item() / (np.pi / 2)

    plt.figure()
    plt.plot(its, vals, label=r'$\theta_{\mathcal{E}}$')
    plt.plot(its, [gt_theta] * len(its), linestyle='--', label=r'Solution 1 $\left(+\frac{\pi}{4}\right)$')
    plt.plot(its, [gt_theta - 1] * len(its), linestyle='--', label=r'Solution 2 $\left(-\frac{\pi}{4}\right)$')
    plt.xlabel('Iterations')
    plt.ylabel('Encoder Angle [π/2 rad]')
    plt.legend()
    plt.savefig('./plot/encoder.png', bbox_inches='tight', format='png')
    plt.show()


def plot_key(frames, key, name=None):
    its, vals = zip(*[(frame.iter_num, getattr(frame, key)) for frame in frames])
    
    plt.figure()
    plt.plot(its, vals, color='k')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.axhline(1, c='gray', ls='--')
    plt.axhline(0, c='gray', ls='--')
    plt.xlabel('Number of Episodes', fontsize=14)
    plt.ylabel(name if name is not None else key.title(), fontsize=14)
    plt.savefig('./plot/alpha.png', bbox_inches='tight', format='png')
    plt.show()
