"""Euclidean manifold."""

from manifolds.base import Manifold
import torch


class Euclidean(Manifold):
    """
    Euclidean Manifold class.

    In every method, the argument c is the curvature (which is 0 everywhere for the Euclidian manifold,
    thus the value given is not used).
    """

    def __init__(self):
        """
        Creates the manifold.
        """
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        """Normalizes a point p. Set vectors with norm larger than 1 to 1."""
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c: float):
        """Squared (Euclidian) distance between pairs of points.
        Sum along last dimension"""
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor, c: float):
        """Converts Euclidean Gradient to Riemannian Gradients. Doesn't do anything for Euclidian space, returns dp."""
        return dp

    def proj(self, p: torch.Tensor, c: float):
        """Projects point p on the manifold. Doesn't do anything for Euclidian space, returns p."""
        return p

    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: float):
        """Projects u on the tangent space of p. Doesn't do anything for Euclidian space, returns u."""
        return u

    def proj_tan0(self, u: torch.Tensor, c: float):
        """Projects u on the tangent space of the origin. Doesn't do anything for Euclidian space, returns u."""
        return u

    def expmap(self, u: torch.Tensor, p: torch.Tensor, c: float):
        """Exponential map of u at point p (map u from tangent space of p to manifold).
        For Euclidian space simply add u to p."""
        return p + u

    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c: float):
        """Logarithmic map of point p1 at point p2 (map p2 from manifold to tangent space of p1).
        For Euclidian space just subtract p1 from p2."""
        return p2 - p1

    def expmap0(self, u, c):
        """Exponential map of u at the origin (map u from tangent space at origin to manifold).
        For Euclidian space just return u."""
        return u

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin (map p from manifold to tangent space at origin).
        For Euclidian space just return p."""
        return p

    def mobius_add(self, x, y, c, dim=-1):
        """Adds point y to x."""
        return x + y

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor, c:float):
        """Multiply matrix m by vector x."""
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w: torch.Tensor, c: float, irange: float=1e-5):
        """Initializes random weigths on the manifold. Input irange determines maximum absolute value of weights.
        Input w is where the new weights are stored."""
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p: torch.Tensor, c: float, u: torch.Tensor, v: torch.Tensor=None, keepdim: bool=False):
        """Inner product for tangent vectors u and v at point x.
        If v is None then take v = u. Input keepdim determines whether to keep the dimensions of the output same as
        the input."""
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y. Doesn't do anything for Euclidian space, returns u."""
        return u

    def ptransp0(self, x, u, c):
        """Parallel transport of u from origin tangent space to x tangent space.
        This does not make sense (should be u) but it is never used."""
        return x + u
