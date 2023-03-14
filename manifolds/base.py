"""Base manifold."""

import torch
from torch.nn import Parameter


class Manifold(object):
    """
    Abstract class to define operations on a manifold.

    In all methods, argument "c" is the value of the curvature, = -1 / K.
    """

    def __init__(self):
        """
        Creates the manifold.
        """
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c: float):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients.
        Unclear what this does but it is only used in Euclidian and Poincare."""
        raise NotImplementedError

    def proj(self, p: torch.Tensor, c: float):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: float):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u: torch.Tensor, c: float):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u: torch.Tensor, p: torch.Tensor, c: float):
        """Exponential map of u at point p (map u from tangent space of p to manifold)."""
        raise NotImplementedError

    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c: float):
        """Logarithmic map of point p1 at point p2 (map p2 from manifold to tangent space of p1)."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin (map u from tangent space at origin to manifold)."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin (map p from manifold to tangent space at origin)."""
        raise NotImplementedError

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c: float, dim=-1):
        """Adds point y to x."""
        raise NotImplementedError

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor, c: float):
        """Performs hyperbolic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w: torch.Tensor, c: float, irange: float=1e-5):
        """Initializes random weigths on the manifold. Input irange determines maximum absolute value of weights.
        Input w is where the new weights are stored."""
        raise NotImplementedError

    def inner(self, p: torch.Tensor, c: float, u: torch.Tensor, v: torch.Tensor=None, keepdim: bool=False):
        """Inner product for tangent vectors u and v at point x.
        If v is None then take v = u. Input keepdim determines whether to keep the dimensions of the output same as
        the input."""
        raise NotImplementedError

    def ptransp(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, c: float):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x: torch.Tensor, u: torch.Tensor, c: float):
        """Parallel transport of u from origin tangent space to x tangent space."""
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.

    Presumably for training curvature.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
