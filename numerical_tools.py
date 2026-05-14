import numpy as np
from numpy.polynomial.hermite import hermgauss
import math
from numba import njit

def uniform_nodes(count, low, high):
    return np.linspace(low, high, int(count), dtype=float)


def cartesian_grid(coords):
    meshes = np.meshgrid(*coords, indexing="ij")
    return np.column_stack([mesh.reshape(-1) for mesh in meshes])


def gauss_hermite_multinormal(n, mu, var):
    n = np.asarray(n, dtype=int)
    nodes, weights = [], []
    for i in range(n.size):
        x_i, w_i = hermgauss(int(n[i]))
        nodes.append(x_i * math.sqrt(2.0))
        weights.append(w_i / math.sqrt(math.pi))
    x = cartesian_grid([np.asarray(v, dtype=float) for v in nodes])
    w = np.asarray(weights[0], dtype=float)
    for wi in weights[1:]:
        w = np.kron(w, np.asarray(wi, dtype=float))
    return x @ np.linalg.cholesky(var) + mu.reshape(1, -1), w


def linear_spline_basis(n, a, b, x):
    x = np.asarray(x, dtype=float).reshape(-1)
    grid = np.linspace(a, b, int(n), dtype=float)
    h = (b - a) / (n - 1)
    basis = np.zeros((x.size, int(n)), dtype=float)
    idx = np.floor((x - a) / h).astype(int)
    idx = np.clip(idx, 0, int(n) - 2)
    left = grid[idx]
    right = grid[idx + 1]
    span = np.maximum(right - left, 1e-12)
    w_right = np.clip((x - left) / span, 0.0, 1.0)
    rows = np.arange(x.size)
    basis[rows, idx] = 1.0 - w_right
    basis[rows, idx + 1] = w_right
    return basis


def apply_inverse_basis_chain(bases, coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    z = coeffs.T.copy()
    rows_total = 1
    for basis in bases:
        basis = np.asarray(basis, dtype=float)
        ni = basis.shape[1]
        m = z.size // ni
        z = z.reshape(m, ni)
        z = basis @ z.T
        rows_total *= basis.shape[0]
    return z.reshape(rows_total, coeffs.shape[1])


def rowwise_kronecker(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], a.shape[1] * b.shape[1])


def tensor_basis_interpolate(phi, coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    out = np.asarray(phi[-1], dtype=float)
    for i in range(len(phi) - 2, -1, -1):
        out = rowwise_kronecker(np.asarray(phi[i], dtype=float), out)
    return out @ coeffs

@njit(cache=True)
def utility(w, theta, b=0.0, CRRA=True):
    """CRRA or CARA utility with optional kinked linear extension below b."""
    w_array = np.asarray(w, dtype=np.float64).reshape(-1)

    if not CRRA:
        # CARA utility: U(w) = 1-exp(-theta * w)
        if theta <= 0:
            raise ValueError("theta must be positive for CARA utility")
        
        if b > 0:
            u_b = 1.0 - np.exp(-theta * b)
            slope = theta * np.exp(-theta * b)
            values = np.where(
                w_array >= b,
                1.0 - np.exp(-theta * w_array),
                u_b + slope * (w_array - b),
            )
        else:
            values = 1.0 - np.exp(-theta * w_array)
        
        return values

    # CRRA utility
    if np.isclose(theta, 0.0):
        values = w_array
    elif np.isclose(theta, 1.0):
        if b > 0:
            values = np.where(
                w_array >= b,
                np.log(np.maximum(w_array, 1e-12)),
                np.log(b) * w_array,
            )
        else:
            values = np.log(np.maximum(w_array, 1e-12))
    else:
        safe_w = np.maximum(w_array, 1e-12)
        crra = (safe_w ** (1 - theta) - 1) / (1 - theta)
        if b > 0:
            u_b = (b ** (1 - theta) - 1) / (1 - theta)
            values = np.where(w_array >= b, crra, u_b * w_array)
        else:
            values = crra

    return values


def model_utility(y, b, theta, crra=True):
    yv = np.asarray(y, dtype=float)
    if not crra:
        if theta <= 0:
            raise ValueError("theta must be positive for CARA utility")
        if b > 0:
            u_b = 1.0 - np.exp(-theta * b)
            slope = theta * np.exp(-theta * b)
            return np.where(yv >= b, 1.0 - np.exp(-theta * yv), u_b + slope * (yv - b))
        return 1.0 - np.exp(-theta * yv)

    if theta == 0:
        return yv
    if theta == 1:
        return np.where(yv >= b, np.log(np.maximum(yv, 1e-12)), (np.log(b) / b) * yv)
    raise ValueError("CRRA utility only implemented for theta=0 or theta=1")


def model_inverse_utility(u, b, theta, crra=True):
    uv = np.asarray(u, dtype=float)
    if not crra:
        if theta <= 0:
            raise ValueError("theta must be positive for CARA utility")

        if b > 0:
            u_b = 1.0 - np.exp(-theta * b)
            slope = theta * np.exp(-theta * b)
            y = b + (uv - u_b) / np.maximum(slope, 1e-300)
            mask = uv >= u_b
            if np.any(mask):
                safe_one_minus_u = np.maximum(1.0 - uv[mask], 1e-300)
                y[mask] = -np.log(safe_one_minus_u) / theta
            return y

        safe_one_minus_u = np.maximum(1.0 - uv, 1e-300)
        return -np.log(safe_one_minus_u) / theta

    if theta == 0:
        return uv
    if theta == 1:
        threshold = np.log(b)
        y = (uv * b) / threshold
        mask = uv >= threshold
        if np.any(mask):
            safe_uv = np.minimum(uv[mask], np.log(np.finfo(float).max))
            y[mask] = np.exp(safe_uv)
        return y
    raise ValueError("CRRA inverse utility only implemented for theta=0 or theta=1")


@njit(cache=True)
def _numba_model_utility_scalar(y, b, theta, crra):
    if not crra:
        if theta <= 0.0:
            return np.nan
        if b > 0.0:
            u_b = 1.0 - np.exp(-theta * b)
            slope = theta * np.exp(-theta * b)
            if y >= b:
                return 1.0 - np.exp(-theta * y)
            return u_b + slope * (y - b)
        return 1.0 - np.exp(-theta * y)

    if theta == 0.0:
        return y
    if theta == 1.0:
        if y >= b:
            return np.log(y if y > 1e-12 else 1e-12)
        return (np.log(b) / b) * y
    return np.nan



@njit(cache=True)
def _numba_model_inverse_utility_scalar(u, b, theta, crra):
    if not crra:
        if theta <= 0.0:
            return np.nan
        if b > 0.0:
            u_b = 1.0 - np.exp(-theta * b)
            slope = theta * np.exp(-theta * b)
            if u >= u_b:
                one_minus_u = 1.0 - u
                if one_minus_u < 1e-300:
                    one_minus_u = 1e-300
                return -np.log(one_minus_u) / theta
            denom = slope if slope > 1e-300 else 1e-300
            return b + (u - u_b) / denom
        one_minus_u = 1.0 - u
        if one_minus_u < 1e-300:
            one_minus_u = 1e-300
        return -np.log(one_minus_u) / theta

    if theta == 0.0:
        return u
    if theta == 1.0:
        if u >= np.log(b):
            return np.exp(u)
        return (b / np.log(b)) * u
    return np.nan



@njit(cache=True)
def _numba_linear_basis_idx_weight(n, a, b, x):
    h = (b - a) / (n - 1)
    idx = int(np.floor((x - a) / h))
    if idx < 0:
        idx = 0
    elif idx > n - 2:
        idx = n - 2
    left = a + h * idx
    right = a + h * (idx + 1)
    span = right - left
    if span < 1e-12:
        span = 1e-12
    w_right = (x - left) / span
    if w_right < 0.0:
        w_right = 0.0
    elif w_right > 1.0:
        w_right = 1.0
    return idx, w_right


@njit(cache=True)
def _numba_interp4_linear(coeffs, n, smin, smax, x0, x1, x2, x3):
    n0 = int(n[0])
    n1 = int(n[1])
    n2 = int(n[2])
    n3 = int(n[3])

    i0, wr0 = _numba_linear_basis_idx_weight(n0, smin[0], smax[0], x0)
    i1, wr1 = _numba_linear_basis_idx_weight(n1, smin[1], smax[1], x1)
    i2, wr2 = _numba_linear_basis_idx_weight(n2, smin[2], smax[2], x2)
    i3, wr3 = _numba_linear_basis_idx_weight(n3, smin[3], smax[3], x3)

    wl0 = 1.0 - wr0
    wl1 = 1.0 - wr1
    wl2 = 1.0 - wr2
    wl3 = 1.0 - wr3

    n23 = n2 * n3
    n123 = n1 * n23

    out = 0.0
    for d0 in range(2):
        j0 = i0 + d0
        w0 = wl0 if d0 == 0 else wr0
        for d1 in range(2):
            j1 = i1 + d1
            w1 = wl1 if d1 == 0 else wr1
            for d2 in range(2):
                j2 = i2 + d2
                w2 = wl2 if d2 == 0 else wr2
                for d3 in range(2):
                    j3 = i3 + d3
                    w3 = wl3 if d3 == 0 else wr3
                    idx = j0 * n123 + j1 * n23 + j2 * n3 + j3
                    out += (w0 * w1 * w2 * w3) * coeffs[idx]
    return out