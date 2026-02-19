"""student.py

Math 589B Assignment 3 (Autograded)

Implement numerical quadrature routines and polynomial interpolation evaluation.

Rules:
- Do not change the required function names/signatures.
- Do not print from your functions.
- You may use numpy and mpmath.

Tip: Barycentric interpolation is the intended approach for stability.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ============================================================
# Quadrature
# ============================================================

def composite_simpson(f: Callable[[float], float], a: float, b: float, n_panels: int) -> float:
    """Composite Simpson's rule on [a,b] using n_panels panels.

    Each panel uses 2 subintervals, so total subintervals = 2*n_panels.

    Parameters
    ----------
    f : callable
        Function f(x) to integrate (scalar -> scalar).
    a, b : float
        Integration interval endpoints.
    n_panels : int
        Number of Simpson panels (must be positive).

    Returns
    -------
    float
        Approximation to \int_a^b f(x) dx.
    """
    raise NotImplementedError


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n_nodes: int) -> float:
    """Gauss-Legendre quadrature on [a,b] with n_nodes.

    You may use numpy's Legendre utilities.

    Returns
    -------
    float
        Approximation to \int_a^b f(x) dx.
    """
    raise NotImplementedError


def romberg(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Romberg integration on [a,b] up to depth n.

    Return the extrapolated value R[n,n].
    Uses Richardson extrapolation applied to trapezoid refinements.

    Parameters
    ----------
    n : int
        Depth (n>=0). Depth 0 returns the single trapezoid rule.

    Returns
    -------
    float
        R[n,n]
    """
    raise NotImplementedError


# ============================================================
# Interpolation (Runge vs Chebyshev)
# ============================================================

def _barycentric_weights(x_nodes: np.ndarray) -> np.ndarray:
    """Compute barycentric weights for distinct nodes.

    This is O(n^2) which is fine for n up to ~50 in this assignment.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    n = x_nodes.size
    w = np.ones(n, dtype=float)
    for j in range(n):
        diff = x_nodes[j] - np.delete(x_nodes, j)
        w[j] = 1.0 / np.prod(diff)
    return w


def _barycentric_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate barycentric interpolant at x_eval."""
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    w = _barycentric_weights(x_nodes)
    out = np.empty_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        diff = x - x_nodes
        hit = np.where(np.abs(diff) < 1e-14)[0]
        if hit.size:
            out[i] = y_nodes[hit[0]]
        else:
            tmp = w / diff
            out[i] = np.sum(tmp * y_nodes) / np.sum(tmp)
    return out


def equispaced_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant Q_n of f at equispaced nodes on [-1,1]."""
    raise NotImplementedError


def chebyshev_lobatto_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant p_n of f at Chebyshev-Lobatto nodes on [-1,1]."""
    raise NotImplementedError


def poly_integral_from_values(x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    """Compute integral over [-1,1] of the interpolating polynomial through (x_nodes, y_nodes).

    You may recover polynomial coefficients (e.g. solve Vandermonde) for moderate n,
    and integrate term-by-term. Alternatively, construct and integrate in another stable way.

    Returns
    -------
    float
        \int_{-1}^1 P(x) dx, where P interpolates the given data.
    """
    raise NotImplementedError
