import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MDN(nn.Module):
    def __init__(self, input_size, hidden_size, n_gaussians):
        super(MDN, self).__init__()

        self.z_h = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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


def mdn_nll(params, x, mean=True):
    pi, mu, sigma = params
    dist = Normal(loc=mu, scale=sigma)
    log_prob_pi_x = dist.log_prob(x) + torch.log(pi)
    loss = -torch.logsumexp(log_prob_pi_x, dim=1)
    return torch.mean(loss) if mean else loss
