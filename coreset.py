"""
coreset.py -- Gradient-descent coreset construction via MMD minimization.

A coreset is a small, weighted point set C = {(x_i, w_i)} that accurately
approximates a larger dataset or data stream. This module minimizes the Maximum
Mean Discrepancy (MMD) between the coreset and mini-batches drawn from the
target distribution, jointly optimizing both the coreset point locations and
their weights via Adam.

Typical usage
-------------
    import torch
    from kernels import RBFKernel
    from coreset import GDBuilder

    kernel = RBFKernel(sigma=0.5)
    builder = GDBuilder(kernel, steps=1000, lr_x=0.01, lr_w=0.5)

    # data_stream is any iterator that yields (batch_size, d) tensors
    coreset, mmd_history = builder.build(data_stream, n=50, dim=2, verbose=True)

    print(coreset.points_nd.shape)   # (50, 2)
    print(coreset.weights_n.sum())   # ~1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import NamedTuple, Protocol, Iterator, Tuple
from kernels import Kernel
from mmd import stochastic_mmd
from tqdm import tqdm


class Coreset(NamedTuple):
    """A weighted point set representing a compressed summary of a dataset.

    Fields
    ------
    points_nd : Tensor of shape (n, d)
        The n coreset point locations in d-dimensional space.
    weights_n : Tensor of shape (n,)
        Non-negative weights that sum to 1.0, one per coreset point.
        Obtained by applying softmax to the raw weight logits after training.
    """
    points_nd: torch.Tensor
    weights_n: torch.Tensor


class CoresetBuilder(Protocol):
    """Protocol that all coreset builder classes must satisfy.
    """

    def build(
        self,
        data_stream: Iterator[torch.Tensor],
        n: int,
        dim: int,
    ) -> Tuple[Coreset, list]:
        """Construct a coreset of size n from the data stream.

        Args:
            data_stream: An iterator that yields mini-batches of shape (b, dim)
                         drawn from the target distribution.
            n:           Number of points in the output coreset.
            dim:         Dimensionality of each data point.

        Returns:
            A (Coreset, history) tuple where history is a list of per-step
            loss values recorded during optimization.
        """
        ...


class GDBuilder:
    """Builds a coreset by minimizing MMD via gradient descent.

    Joint optimization over coreset point locations and weight logits using
    the Adam optimizer. Each step draws a fresh mini-batch from the data
    stream to obtain a stochastic estimate of the MMD objective.

    Supports EMA-based early stopping to halt training once the smoothed loss
    stops improving.
    """

    def __init__(
        self,
        kernel: Kernel,
        steps: int = 2000,
        lr_x: float = 0.01,
        lr_w: float = 0.5,
        device: torch.device = None,
    ):
        """
        Args:
            kernel:        Kernel instance used to compute the MMD objective.
                           See kernels.py for available options (RBFKernel,
                           Matern32Kernel, IMQKernel).
            steps:         Maximum number of gradient steps to take.
                           Actual steps may be fewer if early stopping triggers.
                           Defaults to 2000.
            lr_x:          Adam learning rate applied to point locations. Defaults to 0.01.
            lr_w:          Adam learning rate applied to weights. Defaults to 0.5.
            device:        torch.device on which tensors are allocated
                           (e.g. torch.device('cuda')). Defaults to None,
                           which uses the PyTorch default.
        """
        self.kernel = kernel
        self.steps = steps
        self.lr_x = lr_x
        self.lr_w = lr_w
        self.device = device

    def build(
        self,
        data_stream: Iterator[torch.Tensor],
        n: int,
        dim: int,
        verbose: bool = False,
        patience: int = 100,
        tol: float = 1e-5,
    ) -> Tuple[Coreset, list]:
        """Run gradient descent to build a coreset of size n.

        Jointly optimizes n coreset point locations (x_nd) and n raw weight
        logits (w_logits_n) by minimizing the stochastic MMD objective. After
        training, weights are normalized via softmax.

        Early stopping is controlled by an exponential moving average (EMA) of
        the per-step MMD. Training halts when the EMA has not improved by more
        than tol for `patience` consecutive steps.

        Args:
            data_stream: Iterator yielding (b, dim) Tensors -- mini-batches
                         from the target distribution. Must produce at least
                         `steps` batches.
            n:           Number of coreset points to optimize.
            dim:         Dimensionality of each data point.
            verbose:     If True, display a tqdm progress bar with live MMD
                         and EMA-MMD values, and print an early-stop message
                         when training halts. Defaults to False.
            patience:    Number of steps without improvement (by more than
                         `tol`) before early stopping. Set to 0 to disable
                         early stopping entirely. Defaults to 100.
            tol:         Minimum EMA-MMD decrease to count as an improvement.
                         Defaults to 1e-5.

        Returns:
            coreset:     Coreset with optimized point locations and normalized
                         weights (sum to 1.0).
            mmd_history: List of per-step raw MMD values (floats), one entry
                         per gradient step taken.
        """
        x_nd = nn.Parameter(torch.rand((n, dim), device=self.device))
        w_logits_n = nn.Parameter(torch.zeros(n, device=self.device))

        optimizer = optim.Adam([
            {'params': [x_nd], 'lr': self.lr_x},
            {'params': [w_logits_n], 'lr': self.lr_w}
        ])

        mmd_history = []
        best_ema_mmd = float('inf')
        no_improve_steps = 0
        ema_mmd = None
        ema_alpha = 0.05

        step_iterator = tqdm(range(self.steps), desc="Optimizing MMD") if verbose else range(self.steps)

        for step in step_iterator:
            z_batch_bd = next(data_stream)

            optimizer.zero_grad()
            mmd = stochastic_mmd(self.kernel, x_nd, w_logits_n, z_batch_bd)
            mmd.backward()
            optimizer.step()

            mmd_val = mmd.item()
            mmd_history.append(mmd_val)
            ema_mmd = ema_alpha * mmd_val + (1 - ema_alpha) * ema_mmd if ema_mmd else mmd_val

            if verbose:
                step_iterator.set_postfix(mmd=f"{mmd_val:.5f}", em_avg_mmd=f"{ema_mmd:.5f}")

            if patience > 0:
                if ema_mmd < best_ema_mmd - tol:
                    best_ema_mmd = ema_mmd
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                if no_improve_steps >= patience:
                    if verbose:
                        tqdm.write(f"Early stopping at step {step} (Smoothed loss flatlined around {ema_mmd:.6f})")
                    break

        w_n = torch.softmax(w_logits_n, dim=0).detach()

        return Coreset(points_nd=x_nd.detach(), weights_n=w_n), mmd_history


class KernelHerdingBuilder:
    def __init__(self, kernel, device=torch.device('cpu')):
        self.kernel = kernel
        self.device = device

    def build(
        self,
        data_stream,
        n: int,
        dim: int,
        candidate_pool_size: int = 25000,
        verbose: bool = False,
        **kwargs # Catch leftover GDBuilder args like patience
    ):
        # 1. Collect a large pool of discrete candidate pixels from the stream
        candidates = []
        collected = 0
        if verbose:
            print(f"Collecting {candidate_pool_size} candidate pixels...")

        while collected < candidate_pool_size:
            batch = next(data_stream)
            candidates.append(batch)
            collected += batch.shape[0]

        candidates = torch.cat(candidates, dim=0)[:candidate_pool_size].to(self.device)

        # 2. Approximate the target distribution's kernel mean mu(x)
        # We do this by pulling fresh batches from the image and averaging the kernel
        if verbose:
            print("Approximating target kernel mean...")

        target_mean = torch.zeros(candidate_pool_size, device=self.device)
        n_mean_batches = 20  # Average over a few batches for a stable approximation

        with torch.no_grad():
            for _ in range(n_mean_batches):
                z_batch = next(data_stream).to(self.device)

                # Compute in chunks to prevent MPS/CUDA Out-Of-Memory errors
                chunk_size = 5000
                for i in range(0, candidate_pool_size, chunk_size):
                    chunk = candidates[i:i+chunk_size]
                    k_mat = self.kernel.gram(chunk, z_batch)
                    target_mean[i:i+chunk_size] += k_mat.mean(dim=1)

            target_mean /= n_mean_batches

        # 3. Execute the greedy Kernel Herding loop
        selected_indices = []
        running_penalty = torch.zeros(candidate_pool_size, device=self.device)

        iterator = tqdm(range(n), desc="Kernel Herding") if verbose else range(n)

        with torch.no_grad():
            for t in iterator:
                if t == 0:
                    scores = target_mean
                else:
                    # Score = target_mean - average kernel similarity to already picked points
                    scores = target_mean - (running_penalty / t)

                # Greedily pick the candidate with the highest score
                best_idx = torch.argmax(scores).item()
                selected_indices.append(best_idx)

                # Update the running penalty for the next iteration
                best_point = candidates[best_idx:best_idx+1]
                new_penalty = self.kernel.gram(candidates, best_point).squeeze(1)
                running_penalty += new_penalty

        selected_points = candidates[selected_indices]

        # Kernel Herding implicitly assumes uniform weights
        weights = torch.ones(n, device=self.device) / n

        return Coreset(selected_points, weights), []