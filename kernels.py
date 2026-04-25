"""
kernels.py -- Positive-definite kernel functions used for MMD-based coreset construction.

All kernels implement the Kernel protocol, which requires a single method:

    gram(X_nd, Y_md) -> Tensor of shape (n, m)

where entry [i, j] equals k(X_nd[i], Y_md[j]).

Available kernels
-----------------
RBFKernel     -- Radial Basis Function (Gaussian) kernel
Matern32Kernel -- Matern 3/2 kernel
IMQKernel     -- Inverse Multi-Quadric kernel
"""

from typing import Protocol
import torch


class Kernel(Protocol):
    """Protocol that all kernel objects must satisfy.

    A kernel k(x, y) is a symmetric, positive-definite function that measures
    similarity between two points. Implementing classes must provide a `gram`
    method that evaluates the kernel for all pairs across two point sets.
    """

    def gram(self, X_nd: torch.Tensor, Y_md: torch.Tensor) -> torch.Tensor:
        """Compute the Gram (kernel) matrix between two point sets.

        Args:
            X_nd: Tensor of shape (n, d) -- n points in d dimensions.
            Y_md: Tensor of shape (m, d) -- m points in d dimensions.

        Returns:
            Tensor of shape (n, m) where entry [i, j] = k(X_nd[i], Y_md[j]).
        """
        ...


class RBFKernel:
    """Radial Basis Function (Gaussian) kernel.

    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    The bandwidth parameter sigma controls the length-scale of the kernel.
    Smaller sigma makes the kernel more sensitive to fine-grained differences;
    larger sigma makes it smoother and longer-ranged.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Bandwidth (length-scale) parameter. Must be positive.
                   Defaults to 1.0.
        """
        self.sigma = sigma

    def gram(self, X_nd: torch.Tensor, Y_md: torch.Tensor) -> torch.Tensor:
        """Compute the RBF Gram matrix.

        Args:
            X_nd: Tensor of shape (n, d).
            Y_md: Tensor of shape (m, d).

        Returns:
            Tensor of shape (n, m) with values in (0, 1].
        """
        sq_dist = torch.cdist(X_nd, Y_md, p=2)**2
        return torch.exp(-sq_dist / (2 * self.sigma ** 2))


class Matern32Kernel:
    """Matern 3/2 kernel.

    k(x, y) = (1 + sqrt(3) * ||x - y|| / sigma) * exp(-sqrt(3) * ||x - y|| / sigma)

    This kernel is once-differentiable, making it a middle ground between the
    infinitely smooth RBF kernel and the rougher Matern 1/2 (Laplacian) kernel.
    It is often a better match for real-world data that is not infinitely smooth.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Length-scale parameter controlling the rate of decay.
                   Must be positive. Defaults to 1.0.
        """
        self.sigma = sigma

    def gram(self, X_nd: torch.Tensor, Y_md: torch.Tensor) -> torch.Tensor:
        """Compute the Matern 3/2 Gram matrix.

        Args:
            X_nd: Tensor of shape (n, d).
            Y_md: Tensor of shape (m, d).

        Returns:
            Tensor of shape (n, m) with values in (0, 1].
        """
        dist = torch.cdist(X_nd, Y_md, p=2)
        scaled_d = (torch.sqrt(torch.tensor(3.0)) * dist) / self.sigma
        return (1.0 + scaled_d) * torch.exp(-scaled_d)


class IMQKernel:
    """Inverse Multi-Quadric (IMQ) kernel.

    k(x, y) = c / sqrt(c^2 + ||x - y||^2)

    The IMQ kernel has heavier tails than the RBF kernel, making it more robust
    to outliers and useful when the data distribution has wide support. The
    parameter c controls the transition between the flat region near the origin
    and the power-law decay in the tails.
    """

    def __init__(self, c: float = 1.0):
        """
        Args:
            c: Scale parameter. Must be positive. Larger values widen the
               flat region around the origin. Defaults to 1.0.
        """
        self.c = c

    def gram(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the IMQ Gram matrix.

        Args:
            X: Tensor of shape (n, d).
            Y: Tensor of shape (m, d).

        Returns:
            Tensor of shape (n, m) with values in (0, 1].
        """
        sq_dist = torch.cdist(X, Y, p=2)**2
        return self.c / torch.sqrt(self.c**2 + sq_dist)

