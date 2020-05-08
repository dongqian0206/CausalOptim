import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features, init_mu=None, init_var=None, eps=1.e-6):
        """
        x:              (n = batch_size, k = n_components, d = feature_size)
        mu:             (1, k, d)
        var:            (1, k, d)
        pi:             (1, k, 1)
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.score = -np.inf
        self.eps = eps

        self.pi = nn.Parameter(torch.ones(1, n_components, 1) * (1. / n_components), requires_grad=False)

        if init_mu is not None:
            assert init_mu.size() == (1, n_components, n_features)
            self.mu = nn.Parameter(init_mu, requires_grad=False)
        else:
            self.mu = nn.Parameter(torch.randn(1, n_components, n_features), requires_grad=False)

        if init_var is not None:
            assert init_var.size() == (1, n_components, n_features)
            self.var = nn.Parameter(init_var, requires_grad=False)
        else:
            self.var = nn.Parameter(torch.ones(1, n_components, n_features), requires_grad=False)

    def forward(self, x):
        """
        Return a tensor (n, k)
        """
        batch_size = x.size(0)
        pi = F.softmax(self.pi.squeeze(-1), dim=-1).expand(batch_size, -1)
        mu = self.mu.squeeze(-1).expand(batch_size, -1)
        sigma = torch.sqrt(self.var).squeeze(-1).expand(batch_size, -1)
        return pi, mu, sigma

    def fit(self, x, n_iters=1000, delta=1e-8):

        if len(x.size()) == 2:
            # (n, d) --> (n, k, d)
            x = x.unsqueeze(1).expand(-1, self.n_components, -1)

        i = 0
        j = np.inf

        while (i <= n_iters) and (j >= delta):

            old_score = self.score
            old_mu = self.mu
            old_var = self.var

            self._em(x)
            self.score = self._score(self.pi, self._p_k(x, self.mu, self.var))

            if (self.score.abs() == float('Inf')) or (self.score == float('nan')):
                self.__init__(self.n_components, self.n_features)

            i += 1
            j = self.score - old_score

            if j <= delta:
                self._update_mu(old_mu)
                self._update_var(old_var)

    def _em(self, x):
        """
        Perform one iteration of the EM
        """
        weights = self._e_step(self.pi, self._p_k(x, self.mu, self.var))
        pi_new, mu_new, var_new = self._m_step(x, weights)

        self._update_pi(pi_new)
        self._update_mu(mu_new)
        self._update_var(var_new)

    def _p_k(self, x, mu, var):
        """
        Return a tensor with size (n, k, 1) indicating the likelihood of the k-th Gaussian
        args:
            x:      (n, k, d)
            mu:     (1, k, d)
            var:    (1, k, d)
        returns:
            p_k:    (n, k, 1)
        """

        # (1, k, d) --> (n, k, d)
        batch_size = x.size(0)
        mu = mu.expand(batch_size, -1, -1)
        var = var.expand(batch_size, -1, -1)

        # (n, k, d) --> (n, k, 1)
        exponent = torch.exp(-0.5 * torch.sum((x - mu) ** 2 / var, 2, keepdim=True))

        factor = torch.rsqrt(((2. * math.pi) ** self.n_features) * torch.prod(var, dim=2, keepdim=True) + self.eps)

        return exponent * factor

    def _e_step(self, pi, p_k):
        """
        args:
            pi:         (1, k, 1)
            p_k:        (n, k, 1)
        returns:
            weights:    (n, k, 1)
        """
        weights = pi * p_k
        return torch.div(weights, torch.sum(weights, 1, keepdim=True) + self.eps)

    def _m_step(self, x, weights):
        """
        args:
            x:          (n, k, d)
            weights:    (n, k, 1)
        returns:
            pi_new:     (1, k, 1)
            mu_new:     (1, k, d)
            var_new:    (1, k, d)
        """
        n_k = torch.sum(weights, 0, keepdim=True)
        # (n, k, 1) --> (1, k, 1)
        pi_new = torch.div(n_k, torch.sum(n_k, 1, keepdim=True) + self.eps)
        # (n, k, d) --> (1, k, d)
        mu_new = torch.div(torch.sum(weights * x, 0, keepdim=True), n_k + self.eps)
        # (n, k, d) --> (1, k, d)
        var_new = torch.div(torch.sum(weights * (x - mu_new) ** 2, 0, keepdim=True), n_k + self.eps)

        return pi_new, mu_new, var_new

    def _score(self, pi, p_k):
        """
        Compute the log-likelihood of K Gaussians
        """
        weights = pi * p_k
        return torch.sum(torch.log(torch.sum(weights, 1) + self.eps))

    def _update_pi(self, pi):
        self.pi.data = pi

    def _update_mu(self, mu):
        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def _update_var(self, var):
        if var.size() == (self.n_components, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.n_components, self.n_features):
            self.var.data = var
