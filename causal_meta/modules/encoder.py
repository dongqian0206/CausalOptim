import torch
import torch.nn as nn


class Rotor(nn.Module):
    def __init__(self, init_theta):
        super(Rotor, self).__init__()

        self.theta = nn.Parameter(torch.tensor(init_theta).float(), requires_grad=True)

    def make_rotmat(self):
        mat = self.theta.new(2, 2)
        mat[0, 0] = self.theta.cos()
        mat[0, 1] = -self.theta.sin()
        mat[1, 0] = self.theta.sin()
        mat[1, 1] = self.theta.cos()
        return mat

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        rotmat = self.make_rotmat()
        xy = torch.matmul(xy, rotmat)
        x, y = xy[:, 0:1], xy[:, 1:2]
        return x, y
