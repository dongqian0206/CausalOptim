import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
from copy import deepcopy
from causal_meta.modules.gmm import GaussianMixture


def logsumexp(a, b):
    min_, max_ = torch.min(a, b), torch.max(a, b)
    return max_ + F.softplus(min_ - max_)


def sample_from_normal(mean, std, nsamples):
    return torch.normal(torch.ones(nsamples, 1).mul_(mean),
                        torch.ones(nsamples, 1).mul_(std))


def train_nll(opt, model, scm, nll, polarity='X2Y', encoder=None, decoder=None):

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    frames = []

    for iter_num in range(opt.num_iters):

        with torch.no_grad():
            x = sample_from_normal(0, 2, opt.nsamples)
            y = scm(x)

        with torch.no_grad():
            if encoder is not None:
                x, y = encoder(x, y)
            if decoder is not None:
                x, y = decoder(x, y)

        if polarity == 'X2Y':
            inputs, targets = x, y
        elif polarity == 'Y2X':
            inputs, targets = y, x
        else:
            raise ValueError('%s does not match any known polarity.' % polarity)

        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = nll(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        frames.append(loss.item())

    return frames


def marginal_nll(opt, inputs, nll):
    gmm = GaussianMixture(opt.gmm_num_components).cuda()
    gmm.fit(inputs)
    with torch.no_grad():
        loss_marginal = nll(gmm(inputs), inputs)
    return loss_marginal


def transfer_eval(opt, model, inputs, targets, nll):
    model = deepcopy(model)
    optimizer = optim.Adam(model.parameters(), opt.finetune_lr)

    loss_marginal = marginal_nll(opt, inputs, nll)

    loss_joint = []

    for _ in range(opt.finetune_num_iters):
        preds = model(inputs)
        loss_conditional = nll(preds, targets)
        optimizer.zero_grad()
        loss_conditional.backward()
        optimizer.step()
        loss_joint.append(loss_conditional.item() + loss_marginal.item())

    return sum(loss_joint)


def train_alpha(opt, model_x2y, model_y2x, scm, alpha, nll, mode='logmix'):

    alpha_optimizer = torch.optim.Adam([alpha], lr=opt.alpha_lr)

    frames = []

    for iter_num in range(1, opt.alpha_num_iters + 1):

        # same mechanism
        with torch.no_grad():
            param = np.random.uniform(-4, 4)
            y_gt = sample_from_normal(param, 2, opt.nsamples)
            x_gt = scm(y_gt)
            # x_gt = sample_from_normal(param, 2, opt.nsamples)
            # y_gt = scm(x_gt)

        x_gt, y_gt = x_gt.cuda(), y_gt.cuda()

        # Evaluate performance
        loss_x2y = transfer_eval(opt, model_x2y, x_gt, y_gt, nll)
        loss_y2x = transfer_eval(opt, model_y2x, y_gt, x_gt, nll)

        # Estimate gradient
        if mode == 'mix':
            # sigmoid(alpha) * (-log_likelihood_1) + (1 - sigmoid(alpha)) * (-log_likelihood_2)
            loss_alpha = torch.sigmoid(alpha) * loss_x2y + (1 - torch.sigmoid(alpha)) * loss_y2x

        else:
            # log \sum_{i=1}^{2} exp[ log(sigma_{i}) + log_likelihood_{i}) ]
            log_alpha, log_1_m_alpha = F.logsigmoid(alpha), F.logsigmoid(-alpha)
            as_lse = logsumexp(log_alpha + loss_x2y, log_1_m_alpha + loss_y2x)

            if mode == 'logsigp':
                loss_alpha = as_lse
            elif mode == 'sigp':
                loss_alpha = as_lse.exp()
            else:
                raise ValueError('%s does not match any known mode.' % mode)

        alpha_optimizer.zero_grad()
        loss_alpha.backward()
        alpha_optimizer.step()

        print('| Iteration: %d | Prob: %.3f | X2Y_Loss: %.3f | Y2X_Loss: %.3f'
              % (iter_num, torch.sigmoid(alpha).item(), loss_x2y, loss_y2x))

        with torch.no_grad():
            frames.append(Namespace(iter_num=iter_num,
                                    sig_alpha=torch.sigmoid(alpha).item(),
                                    loss_x2y=loss_x2y,
                                    loss_y2x=loss_y2x,
                                    loss_alpha=loss_alpha.item()))

    return frames
