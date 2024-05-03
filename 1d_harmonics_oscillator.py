import logging
import argparse
import os
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from kan import KAN

parse = argparse.ArgumentParser()

# === model ===
parse.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'kan'])
parse.add_argument('--n_layers', type=int, default=2)
parse.add_argument('--hidden_dim', type=int, default=5)

# === problem ===
parse.add_argument('--d', type=int, default=2)
parse.add_argument('--w0', type=int, default=10)

# === data ===
parse.add_argument('--n_mesh', type=int, default=1000)
parse.add_argument('--pde_sample', type=int, default=100)

# === train ===
parse.add_argument('--n_step', type=int, default=10000)
parse.add_argument('--lr', type=float, default=1e-2)

# === plot ===
parse.add_argument('--plot_data', action='store_true')

args = parse.parse_args()

# create output root
os.makedirs('output', exist_ok=True)

# init logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('output/1d_harmonics_oscillator_%s_%s_%s_%s_%s.log' % (args.model_type, args.d, args.w0, args.n_layers, args.hidden_dim))
# add file handler
logger.addHandler(file_handler)
# remove the handler of cmd
logger.propagate = False


def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative


class MLP(nn.Module):

    def __init__(self, in_dim=1, out_dim=1, hidden_dim=10, n_layers=2):
        super(MLP, self).__init__()
        _in_dim = in_dim

        # build model
        self.model = list()
        for i in range(n_layers):
            self.model.append(nn.Linear(_in_dim, hidden_dim))
            self.model.append(nn.GELU())
            _in_dim = hidden_dim
        self.model.append(nn.Linear(_in_dim, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class HarmonicOscillator1D:
    """
              d^2 u      du
            m ----- + mu -- + ku = 0
              dt^2       dt

            conditions:
            u (t = 0) = 1
            u'(t = 0) = 0
            m = 1

            exact solution:
            u(t) = exp(-d*t) * (A * cos(w*t) + B * sin(w*t))

            d = mu / 2
            w_0 = sqrt(k)
            w = sqrt(w_0^2 - d^2)
            phi = arctan(-d / w)
        """

    def __init__(self, d=2, w0=20):
        self.min_x = 0
        self.max_x = 1
        self.d = d
        self.w0 = w0
        self.mu = 2 * d
        self.k = w0 ** 2

    def exact_solution(self, input_data):
        w = np.sqrt(self.w0 ** 2 - self.d ** 2)
        phi = np.arctan(-self.d / w)
        A = 1 / (2 * np.cos(phi))

        # check the type of input_x
        input_type = type(input_data)
        if input_type == np.ndarray:
            cos = np.cos(phi + w * input_data)
            exp = np.exp(-self.d * input_data)
            u = exp * 2 * A * cos
        elif input_type == torch.Tensor:
            cos = torch.cos(phi + w * input_data)
            exp = torch.exp(-self.d * input_data)
            u = exp * 2 * A * cos
        else:
            raise ValueError('input_data should be numpy array, but got %s' % input_type)

        return u

    def pde_loss(self, pred_tensor, input_tensor):
        du_dt = fwd_gradients(pred_tensor, input_tensor)[:, 0:1]
        du_dtt = fwd_gradients(du_dt, input_tensor)[:, 0:1]

        pde_loss = torch.mean((du_dtt + self.mu * du_dt + self.k * pred_tensor) ** 2)
        return pde_loss

    def ic_loss(self, pred_tensor, input_tensor):
        du_dt = fwd_gradients(pred_tensor, input_tensor)[:, 0:1]
        ic_loss = torch.mean((pred_tensor - 1) ** 2) + torch.mean(du_dt ** 2)
        return ic_loss


def train():
    loss_list = list()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.n_step,
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1
    )

    pbar = tqdm(range(args.n_step))
    for i in pbar:
        # == pde samples ==
        pde_samples = np.random.uniform(pde.min_x, pde.max_x, args.pde_sample)
        pde_samples = torch.tensor(pde_samples, dtype=torch.float32).reshape(-1, 1)
        pde_samples.requires_grad = True

        # == ic samples ==
        ic_samples = torch.tensor([0.0], dtype=torch.float32).reshape(-1, 1)
        ic_samples.requires_grad = True

        # == forward ==
        pde_pred = model(pde_samples)
        ic_pred = model(ic_samples)

        # == loss ==
        pde_loss = pde.pde_loss(pde_pred, pde_samples)
        ic_loss = pde.ic_loss(ic_pred, ic_samples)
        total_loss = pde_loss + ic_loss * 1e4

        with torch.no_grad():
            l2_loss = torch.mean((model(pde_samples) - pde.exact_solution(pde_samples)) ** 2)

        # == update ==
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        scheduler.step()

        # == log ==
        log_str = 'pde loss: %.4e | ic loss: %.4e | l2: %.4e' % (pde_loss.item(), ic_loss.item(), l2_loss.item())
        pbar.set_description(log_str)
        logger.info(f"[{i}/{args.n_step}] | " + log_str)
        loss_list.append(l2_loss.item())
    return loss_list


if __name__ == '__main__':

    # === fix seed ===
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)

    # === init pde ===
    pde = HarmonicOscillator1D(d=args.d, w0=args.w0)

    # === init samples ===
    # == mesh samples ==
    mesh_points = np.linspace(pde.min_x, pde.max_x, args.n_mesh)
    mesh_exact_solution = pde.exact_solution(mesh_points)

    # == plot samples ==

    # = pde samples =
    pde_samples = np.random.uniform(pde.min_x, pde.max_x, args.pde_sample)

    if args.plot_data:
        plt.plot(mesh_points, mesh_exact_solution, label='exact solution')
        plt.scatter(pde_samples, pde.exact_solution(pde_samples), label='pde samples', c='r', marker='x')
        plt.legend()
        plt.show()

    # === init model ===
    if args.model_type == 'mlp':
        model = MLP(in_dim=1, out_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    elif args.model_type == 'kan':
        model = KAN(width=[1]+[args.hidden_dim]*args.n_layers+[1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)
    else:
        raise ValueError('model type %s is not supported' % args.model_type)

    # === training ===
    l2_loss_list = train()

    # === plot ===

    # == prediction ==
    pred_mesh = model(torch.tensor(mesh_points, dtype=torch.float32).reshape(-1, 1))
    plt.title(f'pred ({args.model_type}) vs exact')
    plt.plot(mesh_points, mesh_exact_solution, label='exact solution')
    plt.plot(mesh_points, pred_mesh.detach().numpy(), label='pred solution', c='r', marker='x')
    plt.legend()
    plt.savefig('output/1d_harmonics_oscillator_%s_%s_%s_%s_%s_pred.png' % (args.model_type, args.d, args.w0, args.n_layers, args.hidden_dim))
    plt.clf()

    # == loss ==
    plt.title('loss')
    # plot with log
    plt.plot(l2_loss_list, label='l2 loss (log)')
    plt.yscale('log')
    plt.savefig('output/1d_harmonics_oscillator_%s_%s_%s_%s_%s_l2_loss.png' % (args.model_type, args.d, args.w0, args.n_layers, args.hidden_dim))

