import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


class RandomSplineSCM(nn.Module): 
    def __init__(self, span=6, num_anchors=10, order=3, range_scale=1.,
                 input_noise=False, output_noise=True):
        super(RandomSplineSCM, self).__init__()

        self._span = span
        self._num_anchors = num_anchors
        self._range_scale = range_scale
        self._x = np.linspace(-span, span, num_anchors)
        self._y = np.random.uniform(-range_scale * span, range_scale * span, size=(num_anchors,))
        self._spline_spec = interpolate.splrep(self._x, self._y, k=order)

        self.input_noise = input_noise
        self.output_noise = output_noise

    def forward(self, x, z=None):
        if z is None:
            z = self.sample(x.size())
        if self.input_noise:
            x = x + z
        _x_np = x.detach().cpu().numpy().squeeze()
        _y_np = interpolate.splev(_x_np, self._spline_spec)
        _y = torch.from_numpy(_y_np).view(-1, 1).float()
        y = _y + z if self.output_noise else _y
        return y

    @staticmethod
    def sample(input_size):
        return torch.normal(torch.zeros(*input_size), torch.ones(*input_size))
    
    def plot(self, x, title='Samples from the SCM', label=None, **kwargs):
        y = self.forward(x)
        plt.figure()
        plt.scatter(x.squeeze().numpy(), y.squeeze().numpy(), marker='+', label=label, **kwargs)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


def generate_data_categorical(num_samples, pi_A, pi_B_A):
    """
    Sample data using ancestral sampling
    x_A ~ Categorical(pi_A)
    x_B ~ Categorical(pi_B_A[x_A])
    """
    N = pi_A.shape[0]
    r = np.arange(N)

    x_A = np.dot(np.random.multinomial(1, pi_A, size=num_samples), r)
    x_Bs = np.zeros((num_samples, N), dtype=np.int64)
    for i in range(num_samples):
        x_Bs[i] = np.random.multinomial(1, pi_B_A[x_A[i]], size=1)
    x_B = np.dot(x_Bs, r)

    return np.vstack((x_A, x_B)).T.astype(np.int64)


def sample_from_normal(mean, std, nsamples, n_features):
    return torch.normal(torch.ones(nsamples, n_features) * mean,
                        torch.ones(nsamples, n_features) * std)
