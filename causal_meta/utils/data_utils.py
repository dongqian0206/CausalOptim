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
