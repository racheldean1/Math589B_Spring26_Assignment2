"""
student.py

Numerical quadrature + interpolation.

We implement:
- Composite Simpson
- Gauss–Legendre quadrature (via numpy leggauss)
- Romberg integration (trapezoid + Richardson extrapolation)
- Polynomial interpolation evaluation at (1) equispaced nodes and (2) Chebyshev–Lobatto nodes
  using barycentric interpolation for stability.
- Integral of an interpolating polynomial from its values at nodes.

Allowed: numpy, mpmath, standard library.
Not allowed: high-level integrators like mpmath.quad / scipy.integrate.quad.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import mpmath as mp


# ===============================================================
# Quadrature routines
# ===============================================================

def composite_simpson(f: Callable[[float], float], a: float, b: float, n_panels: int) -> float:
    """
    Composite Simpson's rule on [a,b] using n_panels panels.
    Each Simpson "panel" covers 2 subintervals, so total subintervals N = 2*n_panels.
    """
    if n_panels <= 0:
        raise ValueError("n_panels must be positive")

    # Total number of subintervals
    N = 2 * n_panels
    h = (b - a) / N

    # Simpson formula:
    # integral ≈ (h/3)[f(x0) + f(xN) + 4*sum(f(x_odd)) + 2*sum(f(x_even interior))]
    total = f(float(a)) + f(float(b))

    # odd indices: 1,3,5,...,N-1
    odd_sum = 0.0
    for i in range(1, N, 2):
        x = a + i * h
        odd_sum += f(float(x))

    # even indices: 2,4,6,...,N-2
    even_sum = 0.0
    for i in range(2, N, 2):
        x = a + i * h
        even_sum += f(float(x))

    total = total + 4.0 * odd_sum + 2.0 * even_sum
    return (h / 3.0) * total


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n_nodes: int) -> float:
    """
    Gauss–Legendre quadrature on [a,b] with n_nodes.
    numpy gives nodes/weights on [-1,1]. We map them to [a,b].
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    # nodes x_hat and weights w_hat for [-1,1]
    x_hat, w_hat = np.polynomial.legendre.leggauss(n_nodes)

    # map [-1,1] -> [a,b]: x = mid + half*x_hat
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * x_hat

    # Gauss quadrature: ∫_a^b f(x) dx ≈ half * Σ w_hat_i f(x_i)
    fx = np.array([f(float(xi)) for xi in x], dtype=float)
    return float(half * np.dot(w_hat, fx))


def romberg(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Romberg integration on [a,b] up to depth n.

    Romberg builds trapezoid approximations T_0, T_1, ..., T_n
    where T_k uses 2^k subintervals, then does Richardson extrapolation
    to accelerate convergence.

    IMPORTANT: the assignment hints that using mpmath for intermediate arithmetic
    helps reduce roundoff. We still call f with float (per the note), but we store
    and extrapolate in mp.mpf to reduce cancellation errors.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    # Use more digits for internal arithmetic (this is not "quad" integration,
    # just higher-precision adds/multiplies).
    mp.mp.dps = 80

    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)

    # Romberg table R[k][j], where:
    #   R[k][0] = trapezoid with 2^k subintervals
    #   R[k][j] = Richardson extrapolation using R[k][j-1] and R[k-1][j-1]
    R = [[mp.mpf("0") for _ in range(n + 1)] for __ in range(n + 1)]

    # Base trapezoid (k=0): one interval
    fa = mp.mpf(f(float(a)))
    fb = mp.mpf(f(float(b)))
    R[0][0] = (b_mp - a_mp) * (fa + fb) / 2

    # Build successive refinements
    for k in range(1, n + 1):
        # step size for trapezoid with 2^k intervals
        h = (b_mp - a_mp) / (2 ** k)

        # When refining from 2^(k-1) to 2^k intervals,
        # we add the new midpoints: a + (2i-1)h, i=1..2^(k-1)
        m = 2 ** (k - 1)
        s_new = mp.mpf("0")
        for i in range(1, m + 1):
            x = a_mp + (2 * i - 1) * h
            s_new += mp.mpf(f(float(x)))  # call f with float, but store as mpf

        # trapezoid update formula
        R[k][0] = mp.mpf("0.5") * R[k - 1][0] + h * s_new

        # Richardson extrapolation:
        # R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1])/(4^j - 1)
        for j in range(1, k + 1):
            factor = mp.mpf(4) ** j
            R[k][j] = R[k][j - 1] + (R[k][j - 1] - R[k - 1][j - 1]) / (factor - 1)

    return float(R[n][n])


# ===============================================================
# Barycentric interpolation helpers
# ===============================================================

def _barycentric_weights(x_nodes: np.ndarray) -> np.ndarray:
    """
    Compute barycentric weights:
        w_j = 1 / Π_{k≠j} (x_j - x_k)

    This is O(n^2) which is fine for moderate n here.
    We compute products in higher precision (mpmath) to reduce overflow/underflow,
    then rescale before converting back to float.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    n = x_nodes.size

    mp.mp.dps = 80
    w_mp = [mp.mpf("1") for _ in range(n)]

    for j in range(n):
        prod = mp.mpf("1")
        xj = mp.mpf(x_nodes[j])
        for k in range(n):
            if k == j:
                continue
            prod *= (xj - mp.mpf(x_nodes[k]))
        w_mp[j] = mp.mpf("1") / prod

    # Scaling weights by a constant doesn't change the barycentric formula
    # (it cancels top/bottom). This keeps float conversion safe.
    maxabs = max(abs(val) for val in w_mp)
    w_mp = [val / maxabs for val in w_mp]

    return np.array([float(val) for val in w_mp], dtype=float)


def barycentric_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate interpolating polynomial using barycentric form:

      p(x) = (Σ w_j y_j/(x-x_j)) / (Σ w_j/(x-x_j))

    If x equals a node x_j, return y_j exactly.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    w = _barycentric_weights(x_nodes)

    out = np.empty_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        diff = x - x_nodes

        # If very close to a node, just return the node value
        hit = np.where(np.abs(diff) < 1e-14)[0]
        if hit.size > 0:
            out[i] = y_nodes[hit[0]]
        else:
            tmp = w / diff
            out[i] = np.sum(tmp * y_nodes) / np.sum(tmp)

    return out


# ===============================================================
# Interpolant evaluation functions
# ===============================================================

def equispaced_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """
    Degree-n interpolant of f built on (n+1) equispaced nodes in [-1,1],
    evaluated at x_eval.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    x_eval = np.asarray(x_eval, dtype=float)

    # Equispaced nodes
    x_nodes = np.linspace(-1.0, 1.0, n + 1, dtype=float)

    # Data values
    y_nodes = np.array([f(float(xi)) for xi in x_nodes], dtype=float)

    # Evaluate barycentric interpolant
    return barycentric_eval(x_nodes, y_nodes, x_eval)


def chebyshev_lobatto_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """
    Degree-n interpolant of f built on Chebyshev–Lobatto nodes:
        x_k = cos(pi*k/n),  k=0,...,n
    evaluated at x_eval.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    x_eval = np.asarray(x_eval, dtype=float)

    # Special case: n=0 gives one node at x=cos(0)=1
    if n == 0:
        c = float(f(1.0))
        return np.full_like(x_eval, c, dtype=float)

    k = np.arange(0, n + 1, dtype=float)
    x_nodes = np.cos(np.pi * k / n).astype(float)

    y_nodes = np.array([f(float(xi)) for xi in x_nodes], dtype=float)
    return barycentric_eval(x_nodes, y_nodes, x_eval)


# ===============================================================
# Integral of interpolating polynomial
# ===============================================================

def poly_integral_from_values(x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    """
    Compute ∫_{-1}^1 p(x) dx where p is the interpolating polynomial through
    (x_nodes, y_nodes).

    Main idea: the integral is linear in the data y, so we can write:
        ∫ p(x) dx = Σ w_i y_i
    where w_i are "interpolatory quadrature weights" for these nodes.

    To find weights w_i, enforce exactness on monomials x^k for k=0..n:
        Σ w_i (x_i)^k = ∫_{-1}^1 x^k dx

    This gives a (n+1)x(n+1) linear system using a Vandermonde matrix.
    Here we solve it with numpy.linalg.solve.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.ndim != 1 or y_nodes.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be 1D")
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have same length")

    m = x_nodes.size
    if m == 0:
        raise ValueError("Need at least one node")

    n = m - 1  # degree

    # Build matrix A where A[k,i] = (x_i)^k for k=0..n and i=0..n
    # Then A @ w = moments
    A = np.zeros((n + 1, n + 1), dtype=float)
    for k in range(n + 1):
        A[k, :] = x_nodes ** k

    # moments b[k] = ∫_{-1}^1 x^k dx
    # = 0 for odd k, = 2/(k+1) for even k
    b = np.zeros(n + 1, dtype=float)
    for k in range(n + 1):
        if k % 2 == 0:
            b[k] = 2.0 / (k + 1)
        else:
            b[k] = 0.0

    # Solve for quadrature weights w
    w = np.linalg.solve(A, b)

    # Integral = Σ w_i y_i
    return float(np.dot(w, y_nodes))
