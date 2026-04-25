"""
mmd.py -- Maximum Mean Discrepancy (MMD) objective for coreset optimization.

MMD measures the distance between two probability distributions P and Q in a
reproducing kernel Hilbert space (RKHS). For a weighted point set C = {(x_i, w_i)}
approximating a target distribution P, the squared MMD is:

    MMD^2(C, P) = sum_{i,j} w_i * w_j * k(x_i, x_j)
                  - 2 * sum_i w_i * E_{z~P}[k(x_i, z)]
                  + E_{z,z'~P}[k(z, z')]

The last term is constant with respect to the coreset parameters, so it is
dropped during optimization. This module provides a stochastic estimator of the
remaining two terms using a mini-batch of samples from P.
"""

import torch
from kernels import Kernel


def stochastic_mmd(
        kernel: Kernel,
        x_nd: torch.Tensor,
        w_n: torch.Tensor,
        z_batch_bd: torch.Tensor
    ) -> torch.Tensor:
    """
    Args:
        kernel:      A Kernel instance used to compute pairwise similarities.
        x_nd:        Coreset points, Tensor of shape (n, d). Typically an
                     nn.Parameter being optimized.
        w_n:         Coreset weights, Tensor of shape (n,). Requires that
                     each w_i >= 0 and sum(w_n) ~ 1
        z_batch_bd:  Mini-batch of data samples from the target distribution,
                     Tensor of shape (b, d).

    Returns:
        Scalar Tensor: stochastic estimate of MMD^2(coreset, data).
        Differentiable with respect to x_nd and w_n.
    """
    k_xx_nn = kernel.gram(x_nd, x_nd)
    coreset_term = torch.matmul(w_n, torch.matmul(k_xx_nn, w_n))

    k_xz_nb = kernel.gram(x_nd, z_batch_bd)
    cross_term = 2.0 * torch.dot(w_n, torch.mean(k_xz_nb, axis=1))

    return coreset_term - cross_term