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

    def forward(self, X, Z=None):
        if Z is None:
            Z = self.sample(X.size())
        if self.input_noise:
            X = X + Z
        _X_np = X.detach().cpu().numpy().squeeze()
        _Y_np = interpolate.splev(_X_np, self._spline_spec)
        _Y = torch.from_numpy(_Y_np).view(-1, 1).float()
        Y = _Y + Z if self.output_noise else _Y
        return Y

    @staticmethod
    def sample(input_size):
        return torch.normal(torch.zeros(*input_size), torch.ones(*input_size))
    
    def plot(self, X, title='Samples from the SCM', label=None, show=True, **kwargs):
        Y = self(X)
        plt.figure()
        plt.scatter(X.squeeze().numpy(), Y.squeeze().numpy(), marker='+', label=label, **kwargs)
        if show:
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
