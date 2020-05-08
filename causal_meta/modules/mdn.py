import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MDN(nn.Module):
    def __init__(self, hidden_size, n_gaussians):
        super(MDN, self).__init__()

        self.z_h = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh()
        )
        self.pi = nn.Linear(hidden_size, n_gaussians)
        self.mu = nn.Linear(hidden_size, n_gaussians)
        self.sigma = nn.Linear(hidden_size, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.pi(z_h), -1)
        mu = self.mu(z_h)
        sigma = torch.exp(self.sigma(z_h))
        return pi, mu, sigma


def mdn_nll(params, y, mean=True):
    pi, mu, sigma = params
    dist = Normal(loc=mu, scale=sigma)
    log_prob_pi_y = torch.log(pi) + dist.log_prob(y)
    loss = -torch.logsumexp(log_prob_pi_y, dim=1)
    return torch.mean(loss) if mean else loss
