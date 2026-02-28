"""student.py

Math 589B Programming Assignment 2 (Autograded)

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
import mpmath as mp


# ===============================================================
# Quadrature
# ===============================================================

def composite_simpson(f: Callable[[float], float], a: float, b: float, n_panels: int) -> float:
    """Composite Simpson's rule on [a,b] using n_panels panels.
    
    Each panel uses 2 subintervals, so total subintervals = 2*n_panels.
    """
    if n_panels <= 0:
        raise ValueError("n_panels must be positive")

    # Total number of subintervals
    N = 2 * n_panels
    h = (b - a) / N

    # integral ≈ (h/3)[f(x0) + f(xN) + 4*sum(f(x_odd)) + 2*sum(f(x_even interior))]:
    total = f(float(a)) + f(float(b))

    # for odd indices for x_odd
    odd_sum = 0.0
    for i in range(1, N, 2):
        x = a + i * h
        odd_sum += f(float(x))

    # for even indices for x_even
    even_sum = 0.0
    for i in range(2, N, 2):
        x = a + i * h
        even_sum += f(float(x))

    total = total + 4.0 * odd_sum + 2.0 * even_sum   # plugging into integral eq.
    return (h / 3.0) * total


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n_nodes: int) -> float:
    """Gauss–Legendre quadrature on [a,b] with n_nodes.
    numpy gives nodes/weights on [-1,1]. We will map them to [a,b].
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    # nodes x_hat and weights w_hat for [-1,1]
    x_hat, w_hat = np.polynomial.legendre.leggauss(n_nodes)

    # mapping [-1,1] -> [a,b]: x = mid + half*x_hat
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * x_hat

    # Gauss quadrature
    fx = np.array([f(float(xi)) for xi in x], dtype=float)
    return float(half * np.dot(w_hat, fx))


def romberg(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Romberg integration on [a,b] up to depth n.

    Return the extrapolated value R[n,n].
    Uses Richardson extrapolation applied to trapezoid refinements.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    #using more digits:
    mp.mp.dps = 80

    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)

    # Romberg table
    R = [[mp.mpf("0") for _ in range(n + 1)] for __ in range(n + 1)]

    # Base trapezoid (k=0)
    fa = mp.mpf(f(float(a)))
    fb = mp.mpf(f(float(b)))
    R[0][0] = (b_mp - a_mp) * (fa + fb) / 2

    # Building successive refinements
    for k in range(1, n + 1):
        h = (b_mp - a_mp) / (2 ** k) # step size for trapezoid with 2^k intervals

        # refining from 2^(k-1) to 2^k intervals:
        m = 2 ** (k - 1)      #add midpoints
        s_new = mp.mpf("0")
        for i in range(1, m + 1):
            x = a_mp + (2 * i - 1) * h
            s_new += mp.mpf(f(float(x)))  # call f with float, but store as mpf

        # trapezoid update formula
        R[k][0] = mp.mpf("0.5") * R[k - 1][0] + h * s_new

        # Richardson extrapolation:
        for j in range(1, k + 1):
            factor = mp.mpf(4) ** j
            R[k][j] = R[k][j - 1] + (R[k][j - 1] - R[k - 1][j - 1]) / (factor - 1)

    return float(R[n][n])


# ===============================================================
# Interpolation (Runge vs Chebyshev)
# ===============================================================

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


def barycentric_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate barycentric interpolant at x_eval."""
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    w = _barycentric_weights(x_nodes)
    out = np.empty_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        diff = x - x_nodes
        hit = np.where(np.abs(diff) < 1e-14)[0]
        if hit.size > 0:
            out[i] = y_nodes[hit[0]]
        else:
            tmp = w / diff
            out[i] = np.sum(tmp * y_nodes) / np.sum(tmp)

    return out


def equispaced_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant Q_n of f at equispaced nodes on [-1,1]."""
    
    if n < 0:
        raise ValueError("n must be >= 0")
    x_eval = np.asarray(x_eval, dtype=float)

    # n+1 equally spaced nodes on [-1,1]
    x_nodes = np.linspace(-1.0, 1.0, n + 1)

    # Sampling f at nodes
    y_nodes = np.array([f(float(x)) for x in x_nodes], dtype=float)

    # Evaluating barycentric interpolant
    return barycentric_eval(x_nodes, y_nodes, x_eval)


def chebyshev_lobatto_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant p_n of f at Chebyshev–Lobatto nodes on [-1,1]"""
    
    if n < 0:
        raise ValueError("n must be >= 0")
    x_eval = np.asarray(x_eval, dtype=float)

    # special case for degree 0 interpolant
    if n == 0:
        value = float(f(1.0))  # only node is cos(0) = 1
        return np.full_like(x_eval, value, dtype=float)

    # Chebyshev–Lobatto nodes:
    k = np.arange(0, n + 1)
    x_nodes = np.cos(np.pi * k / n)

    # Sample f at nodes
    y_nodes = np.array([f(float(x)) for x in x_nodes], dtype=float)

    # Evaluate barycentric interpolant
    return barycentric_eval(x_nodes, y_nodes, x_eval)


def poly_integral_from_values(x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    """Compute integral over [-1,1] of the interpolating polynomial through (x_nodes, y_nodes).

    You may recover polynomial coefficients (e.g. solve Vandermonde) for moderate n,
    and integrate term-by-term. Alternatively, construct and integrate in another stable way.

    Returns
    -------
    float
        \int_{-1}^1 P(x) dx, where P interpolates the given data.
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.ndim != 1 or y_nodes.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be 1D arrays")

    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length")

    n = x_nodes.size - 1  # degree of interpolating polynomial

    # Building Vandermonde-type matrix:
    # A[k,i] = (x_i)^k
    A = np.zeros((n + 1, n + 1), dtype=float)
    for k in range(n + 1):
        A[k, :] = x_nodes ** k

    # RHS: exact integrals of monomials on [-1,1]
    b = np.zeros(n + 1, dtype=float)
    for k in range(n + 1):
        if k % 2 == 0:
            b[k] = 2.0 / (k + 1)   #  integral of x^k for for even k
        else:
            b[k] = 0.0            #  integral of x^k for odd k

    # Solve for quadrature weights
    w = np.linalg.solve(A, b)

    # Integral = sum_i w_i * y_i
    return float(np.dot(w, y_nodes))
