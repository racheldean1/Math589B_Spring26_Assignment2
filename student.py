"""
student.py

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
        Approximation to ∫_a^b f(x) dx.
    """
    # Basic sanity
    if n_panels <= 0:
        raise ValueError("n_panels must be positive")

    # Simpson uses an even number of subintervals: N = 2*n_panels
    N = 2 * n_panels
    h = (b - a) / N

    # Nodes: x_0, x_1, ..., x_N
    # Composite Simpson formula:
    #   ∫ ≈ (h/3) [ f(x0) + f(xN) + 4*sum_{odd i} f(x_i) + 2*sum_{even i, i≠0,N} f(x_i) ]
    s0 = f(float(a)) + f(float(b))

    # Sum odd indices: i = 1,3,5,...,N-1
    s_odd = 0.0
    for i in range(1, N, 2):
        x = a + i * h
        s_odd += f(float(x))

    # Sum even indices: i = 2,4,6,...,N-2
    s_even = 0.0
    for i in range(2, N, 2):
        x = a + i * h
        s_even += f(float(x))

    return (h / 3.0) * (s0 + 4.0 * s_odd + 2.0 * s_even)


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n_nodes: int) -> float:
    """Gauss-Legendre quadrature on [a,b] with n_nodes.

    Uses numpy's Legendre utilities to get nodes/weights on [-1,1],
    then maps to [a,b].

    Returns
    -------
    float
        Approximation to ∫_a^b f(x) dx.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    # Get Gauss-Legendre nodes/weights for [-1, 1]
    x_hat, w_hat = np.polynomial.legendre.leggauss(n_nodes)

    # Affine map from [-1,1] to [a,b]:
    #   x = (b-a)/2 * x_hat + (a+b)/2
    #   dx = (b-a)/2 * d(x_hat)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    x = mid + half * x_hat  # mapped nodes
    fx = np.array([f(float(xi)) for xi in x], dtype=float)

    # Integral ≈ half * sum w_hat * f(mapped nodes)
    return float(half * np.dot(w_hat, fx))


def romberg(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Romberg integration on [a,b] up to depth n.

    Return the extrapolated value R[n,n].
    Uses Richardson extrapolation applied to trapezoid refinements.

    Notes on precision:
    - The autograder note says f will be called with scalar floats.
      So function evaluations are still float-based.
    - BUT we can reduce round-off in the *Romberg table arithmetic*
      by doing the trapezoid summations + extrapolation in mpmath (mpf).

    Parameters
    ----------
    n : int
        Depth (n>=0). Depth 0 returns the single trapezoid rule.

    Returns
    -------
    float
        R[n,n]
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    # Use higher precision for the *internal arithmetic*.
    # This helps the leaderboard case where cancellation/roundoff in the Romberg table matters.
    mp.mp.dps = 80  # "enough" extra precision for stable extrapolation

    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)

    # Allocate Romberg table R with mpf entries.
    # We'll fill only the needed lower triangle.
    R = [[mp.mpf("0") for _ in range(n + 1)] for __ in range(n + 1)]

    # --- Step 1: R[0,0] = trapezoid with 1 panel ---
    fa = mp.mpf(f(float(a)))
    fb = mp.mpf(f(float(b)))
    R[0][0] = (b_mp - a_mp) * (fa + fb) / 2

    # --- Step 2: Build trapezoid refinements and Richardson extrapolation ---
    # Trapezoid refinement relation:
    #   T_k = 1/2 * T_{k-1} + h_k * sum_{i=1}^{2^{k-1}} f(a + (2i-1)h_k)
    # where h_k = (b-a)/2^k and we add the "new" midpoints.
    for k in range(1, n + 1):
        # Step size at level k
        h = (b_mp - a_mp) / (2 ** k)

        # Sum f at the new points (odd indices of the refined grid)
        # new points: a + (2i-1)h for i=1..2^{k-1}
        m = 2 ** (k - 1)
        s_new = mp.mpf("0")
        for i in range(1, m + 1):
            x = a_mp + (2 * i - 1) * h
            s_new += mp.mpf(f(float(x)))  # f is called with float per assignment note

        # Trapezoid update
        R[k][0] = mp.mpf("0.5") * R[k - 1][0] + h * s_new

        # Richardson extrapolation:
        #   R[k,j] = R[k,j-1] + (R[k,j-1] - R[k-1,j-1]) / (4^j - 1)
        for j in range(1, k + 1):
            denom = mp.mpf(4) ** j - 1
            R[k][j] = R[k][j - 1] + (R[k][j - 1] - R[k - 1][j - 1]) / denom

    return float(R[n][n])


# ===============================================================
# Interpolation (Runge vs Chebyshev)
# ===============================================================

def _barycentric_weights(x_nodes: np.ndarray) -> np.ndarray:
    """Compute barycentric weights for distinct nodes.

    We compute w_j = 1 / Π_{k≠j} (x_j - x_k).

    Implementation detail:
    - Direct products in float can under/overflow for moderate n.
    - We compute in mpmath with higher precision, then rescale weights
      (scaling cancels in barycentric evaluation).
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    n = x_nodes.size

    mp.mp.dps = 80
    w_mp = [mp.mpf("1") for _ in range(n)]

    for j in range(n):
        xj = mp.mpf(x_nodes[j])
        prod = mp.mpf("1")
        for k in range(n):
            if k == j:
                continue
            prod *= (xj - mp.mpf(x_nodes[k]))
        w_mp[j] = mp.mpf("1") / prod

    # Rescale to avoid inf/0 when converting to float; scaling cancels in barycentric formula
    maxabs = max([abs(val) for val in w_mp])
    if maxabs == 0:
        # Should not happen if nodes are distinct, but keep it safe.
        return np.ones(n, dtype=float)

    w_scaled = [val / maxabs for val in w_mp]
    return np.array([float(val) for val in w_scaled], dtype=float)


def barycentric_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate barycentric interpolant at x_eval.

    Formula:
      p(x) = (Σ_j w_j * y_j / (x - x_j)) / (Σ_j w_j / (x - x_j)),
    with the convention p(x_j) = y_j.

    This is stable and O(n) per evaluation point once weights are known.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    w = _barycentric_weights(x_nodes)
    out = np.empty_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        diff = x - x_nodes

        # If x coincides with a node, return the corresponding y exactly
        hit = np.where(np.abs(diff) < 1e-14)[0]
        if hit.size:
            out[i] = y_nodes[hit[0]]
        else:
            tmp = w / diff
            out[i] = np.sum(tmp * y_nodes) / np.sum(tmp)

    return out


def equispaced_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant Q_n of f at equispaced nodes on [-1,1] at x_eval.

    Steps:
    1) Build equispaced nodes x_0,...,x_n in [-1,1] (n+1 nodes).
    2) Sample y_j = f(x_j).
    3) Evaluate the barycentric interpolant at x_eval.

    Returns
    -------
    np.ndarray
        Values Q_n(x_eval) (same shape as x_eval).
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    x_eval = np.asarray(x_eval, dtype=float)

    # 1) n+1 equispaced nodes on [-1,1]
    x_nodes = np.linspace(-1.0, 1.0, n + 1, dtype=float)

    # 2) data values at nodes
    y_nodes = np.array([f(float(xi)) for xi in x_nodes], dtype=float)

    # 3) barycentric evaluation
    return barycentric_eval(x_nodes, y_nodes, x_eval)


def chebyshev_lobatto_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant p_n of f at Chebyshev-Lobatto nodes on [-1,1] at x_eval.

    Chebyshev-Lobatto nodes:
      x_k = cos(pi*k/n),  k=0,...,n  (n+1 nodes), with endpoints included.

    Steps:
    1) Build Chebyshev-Lobatto nodes.
    2) Sample y_k = f(x_k).
    3) Evaluate barycentric interpolant at x_eval.

    Returns
    -------
    np.ndarray
        Values p_n(x_eval) (same shape as x_eval).
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    x_eval = np.asarray(x_eval, dtype=float)

    if n == 0:
        # Degree 0 interpolant is just constant f(1) (since only node is x_0 = cos(0)=1)
        c = float(f(1.0))
        return np.full_like(x_eval, c, dtype=float)

    k = np.arange(0, n + 1, dtype=float)

    # 1) Chebyshev-Lobatto nodes on [-1,1]
    x_nodes = np.cos(np.pi * k / n).astype(float)

    # 2) values at nodes
    y_nodes = np.array([f(float(xi)) for xi in x_nodes], dtype=float)

    # 3) barycentric evaluation
    return barycentric_eval(x_nodes, y_nodes, x_eval)


def poly_integral_from_values(x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    """Compute integral over [-1,1] of the interpolating polynomial through (x_nodes, y_nodes).

    Key idea:
    - The integral of the interpolant can be written as a weighted sum of y-values:
        ∫_{-1}^1 p(x) dx = Σ_i w_i * y_i
      where w_i are the *interpolatory quadrature weights* associated to the nodes x_i.

    How we compute weights:
    - Let n = len(x_nodes)-1 (degree n interpolant).
    - Enforce exactness on monomials x^k for k=0..n:
        Σ_i w_i * (x_i)^k = ∫_{-1}^1 x^k dx
    - This gives a linear system (V^T) w = m, where V_{i,k} = x_i^k.

    Numerical stability:
    - Vandermonde systems can be ill-conditioned in float.
    - We solve using mpmath with higher precision, then return float result.

    Returns
    -------
    float
        ∫_{-1}^1 p(x) dx
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.ndim != 1 or y_nodes.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be 1D arrays")
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length")

    m = x_nodes.size
    if m == 0:
        raise ValueError("x_nodes must be nonempty")

    n = m - 1  # degree

    # Use high precision linear algebra for robustness
    mp.mp.dps = 100

    # Build A = V^T where V_{i,k} = x_i^k, so A_{k,i} = x_i^k
    A = mp.matrix(n + 1, n + 1)
    for k in range(n + 1):
        for i in range(n + 1):
            A[k, i] = mp.mpf(x_nodes[i]) ** k

    # Moments m_k = ∫_{-1}^1 x^k dx
    # = 0 for odd k, = 2/(k+1) for even k
    b = mp.matrix(n + 1, 1)
    for k in range(n + 1):
        if k % 2 == 1:
            b[k] = mp.mpf("0")
        else:
            b[k] = mp.mpf("2") / mp.mpf(k + 1)

    # Solve A * w = b for weights w (size n+1)
    w = mp.lu_solve(A, b)

    # Weighted sum of y-values
    total = mp.mpf("0")
    for i in range(n + 1):
        total += w[i] * mp.mpf(y_nodes[i])

    return float(total)
