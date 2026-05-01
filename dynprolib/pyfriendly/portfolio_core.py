from dataclasses import dataclass
from pathlib import Path
import csv
import importlib
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numba import njit

try:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    jax.config.update("jax_enable_x64", True)
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

np.set_printoptions(precision=5, suppress=True)


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


def _read_farmland_prices(path):
    prices = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            prices.append(float(row[1]))
    return np.asarray(prices, dtype=float)


def _read_sp500_values(path):
    values = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            values.append(float(row[1]))
    return np.asarray(values, dtype=float)


def _resolve_data_path(data_dir):
    """Resolve data folder from common locations used in this workspace."""
    candidate = Path(data_dir)
    if candidate.is_absolute():
        return candidate

    candidates = [
        Path.cwd() / candidate,
        Path.cwd() / "Dynamic-Programming-Project" / candidate,
        Path.cwd() / "DynPro 2026" / "Dynamic-Programming-Project" / candidate,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_historical_gross_returns(data_dir="Data"):
    data_path = _resolve_data_path(data_dir)
    farmland_prices = _read_farmland_prices(data_path / "farmland_prices.csv")
    sp500_values = _read_sp500_values(
        data_path / "sp500_inflation_adjusted_yearly_1967_2007.csv"
    )

    farmland_gross = farmland_prices[1:] / farmland_prices[:-1]
    sp500_gross = sp500_values[1:] / sp500_values[:-1]

    n = min(len(farmland_gross), len(sp500_gross))
    return farmland_gross[-n:], sp500_gross[-n:]


@dataclass
class PortfolioConfig:
    n: tuple[int, int, int, int] = (7, 7, 5, 21)
    q: int = 41
    qf: int = 21
    node: str = "nodeunif"
    m: tuple[int, int] = (5, 5)
    beta: float = 0.96
    CRRA: bool = True
    theta: float = 0.0
    b: float = 60_000.0
    T: int = 19
    t1: int = 1
    beta0: float = 1.51181
    beta1: float = 0.742391
    alpha0: float = 0.215
    alpha1: float = 0.908361
    alpha2: float = 0.079432
    Ee: tuple[float, float] = (0.0, 0.0)
    var_cov: tuple[tuple[float, float], tuple[float, float]] = ((0.030186, 0.0), (0.0, 0.017292))
    cost: float = 231.0
    sminv: tuple[float, float, float, float] = (230.0, 1010.0, 400.0, 0.0)
    smaxv: tuple[float, float, float, float] = (540.0, 2840.0, 2000.0, 6_000_000.0)
    rp: float = 0.03
    rn: float = 0.06
    tcs: float = 0.06
    tcb: float = 0.01
    fme: float = 300.0
    fds: float = 0.07
    dau: float = 0.7
    wealth_min: float = 100.0
    wealth_max: float = 5_000.0
    wealth_size: int = 250
    action_grid_step: float = 0.1
    risk_free_gross: float | None = None
    data_dir: str = "Data"


class PortfolioChoiceModel:
    """Notebook-friendly model aligned with the Dynamic-Programming-Project style."""

    def __init__(self, config=None, farmland_gross_returns=None, sp500_gross_returns=None, use_jax=False):
        self.config = config or PortfolioConfig()

        self.beta = float(self.config.beta)
        self.CRRA = bool(self.config.CRRA)
        self.theta = float(self.config.theta)
        self.b = float(self.config.b)
        if self.config.risk_free_gross is None:
            self.risk_free_gross = 1.0 + float(self.config.rp)
        else:
            self.risk_free_gross = float(self.config.risk_free_gross)

        self.wealth_grid = np.linspace(
            self.config.wealth_min,
            self.config.wealth_max,
            self.config.wealth_size,
        )

        if farmland_gross_returns is None or sp500_gross_returns is None:
            default_farmland, default_sp500 = load_historical_gross_returns(
                self.config.data_dir
            )
            farmland_gross_returns = (
                default_farmland if farmland_gross_returns is None else farmland_gross_returns
            )
            sp500_gross_returns = (
                default_sp500 if sp500_gross_returns is None else sp500_gross_returns
            )

        self.farmland_gross_returns = np.asarray(farmland_gross_returns, dtype=float)
        self.sp500_gross_returns = np.asarray(sp500_gross_returns, dtype=float)

        if self.farmland_gross_returns.shape != self.sp500_gross_returns.shape:
            raise ValueError("Return arrays must have the same shape")

        self.actions = self._build_action_grid(self.config.action_grid_step)
        self.state_size = self.wealth_grid.size
        self.action_size = self.actions.shape[0]
        self.use_jax = bool(use_jax)
        if self.use_jax and jax is None:
            raise ImportError("JAX is not installed. Install jax/jaxlib or set use_jax=False.")

        # Precompute action return matrix for fast Bellman updates.
        self._action_returns = (
            self.actions[:, 0][:, None] * self.risk_free_gross
            + self.actions[:, 1][:, None] * self.farmland_gross_returns[None, :]
            + self.actions[:, 2][:, None] * self.sp500_gross_returns[None, :]
        )

        if self.use_jax:
            self._init_jax_backend()

    def _init_jax_backend(self):
        self._wealth_grid_jax = jnp.asarray(self.wealth_grid)
        self._action_returns_jax = jnp.asarray(self._action_returns)
        self._flow_utility_jax = jnp.asarray(
            utility(self.wealth_grid, self.theta, self.b, self.CRRA)
        )
        self._beta_jax = jnp.asarray(self.beta)

        wealth_min = float(self.wealth_grid[0])
        wealth_max = float(self.wealth_grid[-1])

        def bellman_jit(v):
            # Shape: (W, A, S)
            next_wealth = (
                self._wealth_grid_jax[:, None, None] * self._action_returns_jax[None, :, :]
            )
            next_wealth = jnp.clip(next_wealth, wealth_min, wealth_max)
            interp = jnp.interp(
                next_wealth.reshape(-1),
                self._wealth_grid_jax,
                v,
            ).reshape(next_wealth.shape)
            continuation = jnp.mean(interp, axis=2)
            value_candidates = self._flow_utility_jax[:, None] + self._beta_jax * continuation
            policy_idx = jnp.argmax(value_candidates, axis=1)
            v_next = jnp.max(value_candidates, axis=1)
            return v_next, policy_idx

        self._bellman_jit = jax.jit(bellman_jit)

    @staticmethod
    def _build_action_grid(step):
        if step <= 0 or step > 1:
            raise ValueError("action_grid_step must be in (0, 1]")

        farmland_weights = np.arange(0.0, 1.0 + 1e-12, step)
        actions = []
        for w_farmland in farmland_weights:
            sp500_weights = np.arange(0.0, 1.0 - w_farmland + 1e-12, step)
            for w_sp500 in sp500_weights:
                w_risk_free = 1.0 - w_farmland - w_sp500
                actions.append((w_risk_free, w_farmland, w_sp500))
        return np.asarray(actions, dtype=float)

    def _portfolio_gross_returns(self, action):
        return (
            action[0] * self.risk_free_gross
            + action[1] * self.farmland_gross_returns
            + action[2] * self.sp500_gross_returns
        )
    def expected_continuation_value(self, V, action):
        portfolio_returns = self._portfolio_gross_returns(action)
        next_wealth = self.wealth_grid[:, None] * portfolio_returns[None, :]
        next_wealth = np.clip(next_wealth, self.wealth_grid[0], self.wealth_grid[-1])

        interpolated = np.empty_like(next_wealth)
        for col in range(next_wealth.shape[1]):
            interpolated[:, col] = np.interp(next_wealth[:, col], self.wealth_grid, V)
        return interpolated.mean(axis=1)
    def bellman_operator(self, V):
        flow_utility = utility(self.wealth_grid, self.theta, self.b, self.CRRA)
        value_candidates = np.empty((self.state_size, self.action_size))

        for a_idx, action in enumerate(self.actions):
            continuation = self.expected_continuation_value(V, action)
            value_candidates[:, a_idx] = flow_utility + self.beta * continuation

        policy_idx = np.argmax(value_candidates, axis=1)
        V_next = value_candidates[np.arange(self.state_size), policy_idx]
        return V_next, policy_idx

    def bellman_operator_jax(self, V):
        if not self.use_jax:
            raise RuntimeError("JAX backend is disabled. Initialize with use_jax=True.")
        v_next, policy_idx = self._bellman_jit(jnp.asarray(V))
        return np.asarray(v_next), np.asarray(policy_idx, dtype=int)
    def value_iteration(self, tol=1e-6, max_iter=2_000, return_history=False):
        V = np.zeros(self.state_size)
        history = []

        for it in range(max_iter):
            if self.use_jax:
                V_next, policy_idx = self.bellman_operator_jax(V)
            else:
                V_next, policy_idx = self.bellman_operator(V)
            error = np.max(np.abs(V_next - V))
            if return_history:
                history.append(error)
            if error < tol:
                if return_history:
                    return V_next, policy_idx, it + 1, history
                return V_next, policy_idx, it + 1
            V = V_next

        if return_history:
            return V, policy_idx, max_iter, history
        return V, policy_idx, max_iter
    def evaluate_policy(self, policy_idx, tol=1e-8, max_iter=3_000):
        V = np.zeros(self.state_size)
        flow_utility = utility(self.wealth_grid, self.theta, self.b, self.CRRA)

        for _ in range(max_iter):
            V_next = np.empty_like(V)
            for s, a_idx in enumerate(policy_idx):
                cont = self.expected_continuation_value(V, self.actions[a_idx])
                V_next[s] = flow_utility[s] + self.beta * cont[s]
            if np.max(np.abs(V_next - V)) < tol:
                return V_next
            V = V_next

        return V
    def policy_iteration(self, tol=1e-8, max_iter=200):
        policy_idx = np.zeros(self.state_size, dtype=int)

        for it in range(max_iter):
            V_policy = self.evaluate_policy(policy_idx, tol=tol)
            _, greedy_policy = self.bellman_operator(V_policy)
            if np.array_equal(greedy_policy, policy_idx):
                return V_policy, policy_idx, it + 1
            policy_idx = greedy_policy

        V_policy = self.evaluate_policy(policy_idx, tol=tol)
        return V_policy, policy_idx, max_iter

    def chosen_actions(self, policy_idx):
        return self.actions[np.asarray(policy_idx, dtype=int)]


def plot_value_function(model, V, ax=None, title="Value Function"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(model.wealth_grid, V, linewidth=2)
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_policy_function(model, policy_idx, ax=None, title="Policy Function"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    chosen = model.chosen_actions(policy_idx)
    ax.plot(model.wealth_grid, chosen[:, 1], label="Farmland weight", linewidth=2)
    ax.plot(model.wealth_grid, chosen[:, 2], label="S&P500 weight", linewidth=2)
    ax.plot(model.wealth_grid, chosen[:, 0], label="Risk-free weight", linewidth=2)
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Portfolio share")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def choose_value_maximizer(cfg):
    return "vmaxh1" if cfg.qf == 0 else "vmaxh2"


def build_model_arrays(cfg):
    n = np.array(cfg.n, dtype=int)
    sminv = np.array(cfg.sminv, dtype=float)
    smaxv = np.array(cfg.smaxv, dtype=float)
    return {
        "n": n,
        "m": np.array(cfg.m, dtype=int),
        "Ee": np.array(cfg.Ee, dtype=float),
        "VarCov": np.array(cfg.var_cov, dtype=float),
        "sminv": sminv,
        "smaxv": smaxv,
        "smin": sminv.copy(),
        "smax": smaxv.copy(),
    }


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


def feasible_trade_bounds(state, cfg, sminv, smaxv):
    x_lower = sminv[2] - state[:, 2]
    x_upper_land = smaxv[2] - state[:, 2]
    sell_price = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    buy_price = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
    x_upper_debt = np.maximum(
        0.0,
        (state[:, 3] - (1.0 - cfg.dau) * sell_price * state[:, 2]) / (buy_price - cfg.dau * sell_price),
    )
    return x_lower, np.minimum(x_upper_land, x_upper_debt)


def transition_next_state(state, trade_x, shocks, cfg, sminv, smaxv):
    next_state = np.zeros_like(state, dtype=float)
    growth_r = np.exp(cfg.beta0 + cfg.beta1 * np.log(state[:, 0]) + shocks[:, 0])
    growth_p = np.exp(cfg.alpha0 + cfg.alpha1 * np.log(state[:, 1]) + cfg.alpha2 * np.log(state[:, 0]) + shocks[:, 1])
    next_state[:, 0] = np.clip(growth_r, sminv[0], smaxv[0])
    next_state[:, 1] = np.clip(growth_p, sminv[1], smaxv[1])
    next_state[:, 2] = state[:, 2] + trade_x

    buy_price = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
    sell_mask = trade_x < 0
    buy_price[sell_mask] = (1.0 - cfg.tcs) * state[sell_mask, 1] + (1.0 - cfg.fds) * cfg.fme
    sell_price_now = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    sell_price_next = (1.0 - cfg.tcs) * next_state[:, 1] + (1.0 - cfg.fds) * cfg.fme

    equity_now = state[:, 3] - sell_price_now * state[:, 2]
    base_rate = np.full(state.shape[0], cfg.rp, dtype=float)
    low_land = next_state[:, 2] < 1.0

    cash_after_trade = equity_now - buy_price * trade_x
    rate_low_land = base_rate.copy()
    rate_low_land[cash_after_trade < 0] = cfg.rn
    wealth_low_land = (1.0 + rate_low_land) * cash_after_trade

    cash_after_cost = equity_now - buy_price * trade_x - cfg.cost * next_state[:, 2]
    rate_high_land = base_rate.copy()
    rate_high_land[cash_after_cost < 0] = cfg.rn
    operating_assets = (1.0 + rate_high_land) * cash_after_cost + next_state[:, 0] * next_state[:, 2]
    wealth_high_land = operating_assets + sell_price_next * next_state[:, 2]

    next_state[:, 3] = np.where(low_land, wealth_low_land, wealth_high_land)
    return next_state


def evaluate_candidate_trade(state, coeffs, t, trade_x, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    nn = state.shape[0]
    value = np.zeros(nn, dtype=float)
    n_shocks = 1 if trade_x is None else shock_nodes.shape[0]
    use_crra = bool(getattr(cfg, "CRRA", True))

    for k in range(n_shocks):
        if trade_x is None:
            g = state.copy()
            wk = 1.0
        else:
            ek = np.tile(shock_nodes[k], (nn, 1))
            g = transition_next_state(state, trade_x, ek, cfg, sminv, smaxv)
            wk = float(shock_weights[k])

        current_val = np.zeros(nn, dtype=float)
        low_land = g[:, 2] < 1.0
        g4 = g[:, 3]
        if np.any(low_land):
            wealth = g4[low_land]
            rate = np.full(wealth.size, cfg.rp, dtype=float)
            rate[wealth < 0] = cfg.rn
            current_val[low_land] = model_utility(
                ((1.0 + rate) ** (cfg.T - t)) * wealth,
                cfg.b,
                cfg.theta,
                use_crra,
            )

        high_land = ~low_land
        if np.any(high_land):
            idx = np.where(high_land)[0]
            if coeffs is None or coeffs.size == 0:
                current_val[idx] = model_utility(g4[idx], cfg.b, cfg.theta, use_crra)
            else:
                idx_pos = idx[g4[idx] > 0]
                if idx_pos.size > 0:
                    g4_cap = np.minimum(g4[idx_pos], smaxv[3])
                    residual = g4[idx_pos] - g4_cap
                    g_eval = g[idx_pos].copy()
                    g_eval[:, 3] = g4_cap
                    phi = [
                        linear_spline_basis(int(n[d]), float(smin[d]), float(smax[d]), g_eval[:, d])
                        for d in range(len(n))
                    ]
                    v_interp = tensor_basis_interpolate(phi, coeffs.reshape(-1, 1)).reshape(-1)
                    wealth_from_interp = model_inverse_utility(v_interp, cfg.b, cfg.theta, use_crra)
                    over_cap = g4[idx_pos] > smaxv[3]
                    if np.any(over_cap):
                        wealth_from_interp[over_cap] = model_utility(
                            wealth_from_interp[over_cap] + ((1.0 + cfg.rp) ** (cfg.T - t)) * residual[over_cap],
                            cfg.b,
                            cfg.theta,
                            use_crra,
                        )
                    current_val[idx_pos] = v_interp
                    if np.any(over_cap):
                        current_val[idx_pos[over_cap]] = wealth_from_interp[over_cap]
                idx_nonpos = idx[g4[idx] <= 0]
                if idx_nonpos.size > 0:
                    current_val[idx_nonpos] = model_utility(
                        ((1.0 + cfg.rn) ** (cfg.T - t)) * g4[idx_nonpos],
                        cfg.b,
                        cfg.theta,
                        use_crra,
                    )

        value += current_val * wk
    return value


def evaluate_trade_grid_with_endpoints(state, coeffs, t, x_low, x_high, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    nn = state.shape[0]
    x_candidates = np.zeros((cfg.q + 2, nn), dtype=float)
    v_candidates = np.zeros((cfg.q + 2, nn), dtype=float)
    gap = (x_high - x_low) / (cfg.q - 1)

    for qi in range(cfg.q + 2):
        if qi == 0:
            x_q = np.zeros(nn, dtype=float)
        elif qi == 1:
            x_q = -state[:, 2]
        else:
            x_q = x_low + gap * (qi - 2)
        v_q = evaluate_candidate_trade(
            state, coeffs, t, x_q, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
        )
        x_candidates[qi, :] = x_q
        v_candidates[qi, :] = v_q
    return v_candidates, x_candidates


def evaluate_trade_grid_coarse(state, coeffs, t, x_low, x_high, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    nn = state.shape[0]
    x_candidates = np.zeros((cfg.q, nn), dtype=float)
    v_candidates = np.zeros((cfg.q, nn), dtype=float)
    gap = (x_high - x_low) / (cfg.q - 1)

    for qi in range(cfg.q):
        x_q = x_low + gap * qi
        v_q = evaluate_candidate_trade(
            state, coeffs, t, x_q, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
        )
        x_candidates[qi, :] = x_q
        v_candidates[qi, :] = v_q
    return v_candidates, x_candidates


def evaluate_trade_grid_fine(state, coeffs, t, x_low, x_high, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    nn = state.shape[0]
    x_candidates = np.zeros((cfg.qf + 2, nn), dtype=float)
    v_candidates = np.zeros((cfg.qf + 2, nn), dtype=float)
    gap = (x_high - x_low) / (cfg.qf - 1)

    for qi in range(cfg.qf + 2):
        if qi == 0:
            x_q = np.zeros(nn, dtype=float)
        elif qi == 1:
            x_q = -state[:, 2]
        else:
            x_q = x_low + gap * (qi - 2)
        v_q = evaluate_candidate_trade(
            state, coeffs, t, x_q, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
        )
        x_candidates[qi, :] = x_q
        v_candidates[qi, :] = v_q
    return v_candidates, x_candidates


def maximize_value_v1(state, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    x_low, x_high = feasible_trade_bounds(state, cfg, sminv, smaxv)
    v_grid, x_grid = evaluate_trade_grid_with_endpoints(
        state, coeffs, t, x_low, x_high, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
    )
    argmax_idx = np.argmax(v_grid, axis=0)
    j = np.arange(state.shape[0])
    return x_grid[argmax_idx, j], v_grid[argmax_idx, j]


def maximize_value_v2(state, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    x_low, x_high = feasible_trade_bounds(state, cfg, sminv, smaxv)
    v_coarse, x_coarse = evaluate_trade_grid_coarse(
        state, coeffs, t, x_low, x_high, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
    )
    argmax_coarse = np.argmax(v_coarse, axis=0)
    j = np.arange(state.shape[0])
    x_star_coarse = x_coarse[argmax_coarse, j]

    refine_span = 1600.0 / (cfg.q - 1)
    x_refine_low = np.maximum(x_low, x_star_coarse - refine_span)
    x_refine_high = np.minimum(x_high, x_star_coarse + refine_span)

    v_fine, x_fine = evaluate_trade_grid_fine(
        state, coeffs, t, x_refine_low, x_refine_high, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
    )
    argmax_fine = np.argmax(v_fine, axis=0)
    return x_fine[argmax_fine, j], v_fine[argmax_fine, j]


def get_value_maximizer(name):
    return maximize_value_v1 if name == "vmaxh1" else maximize_value_v2


def _linear_spline_basis_jax(n, a, b, x):
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    grid = jnp.linspace(a, b, int(n), dtype=jnp.float64)
    h = (b - a) / (n - 1)
    idx = jnp.floor((x - a) / h).astype(jnp.int32)
    idx = jnp.clip(idx, 0, int(n) - 2)
    left = grid[idx]
    right = grid[idx + 1]
    span = jnp.maximum(right - left, 1e-12)
    w_right = jnp.clip((x - left) / span, 0.0, 1.0)
    basis = jnp.zeros((x.size, int(n)), dtype=jnp.float64)
    rows = jnp.arange(x.size)
    basis = basis.at[rows, idx].set(1.0 - w_right)
    basis = basis.at[rows, idx + 1].set(w_right)
    return basis


def _rowwise_kronecker_jax(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], a.shape[1] * b.shape[1])


def _tensor_basis_interpolate_jax(phi, coeffs):
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    out = jnp.asarray(phi[-1], dtype=jnp.float64)
    for i in range(len(phi) - 2, -1, -1):
        out = _rowwise_kronecker_jax(jnp.asarray(phi[i], dtype=jnp.float64), out)
    return out @ coeffs


def _model_utility_jax(y, b, theta, crra=True):
    yv = jnp.asarray(y, dtype=jnp.float64)
    if not crra:
        if theta <= 0:
            raise ValueError("theta must be positive for CARA utility")
        if b > 0:
            u_b = 1.0 - jnp.exp(-theta * b)
            slope = theta * jnp.exp(-theta * b)
            return jnp.where(yv >= b, 1.0 - jnp.exp(-theta * yv), u_b + slope * (yv - b))
        return 1.0 - jnp.exp(-theta * yv)

    if theta == 0:
        return yv
    if theta == 1:
        return jnp.where(yv >= b, jnp.log(jnp.maximum(yv, 1e-12)), (jnp.log(b) / b) * yv)
    raise ValueError("CRRA utility only implemented for theta=0 or theta=1")


def _model_inverse_utility_jax(u, b, theta, crra=True):
    uv = jnp.asarray(u, dtype=jnp.float64)
    if not crra:
        if theta <= 0:
            raise ValueError("theta must be positive for CARA utility")
        if b > 0:
            u_b = 1.0 - jnp.exp(-theta * b)
            slope = theta * jnp.exp(-theta * b)
            y = b + (uv - u_b) / jnp.maximum(slope, 1e-300)
            mask = uv >= u_b
            safe_one_minus_u = jnp.maximum(1.0 - uv, 1e-300)
            return jnp.where(mask, -jnp.log(safe_one_minus_u) / theta, y)
        safe_one_minus_u = jnp.maximum(1.0 - uv, 1e-300)
        return -jnp.log(safe_one_minus_u) / theta

    if theta == 0:
        return uv
    if theta == 1:
        threshold = jnp.log(b)
        y = (uv * b) / threshold
        mask = uv >= threshold
        safe_uv = jnp.minimum(uv, jnp.log(jnp.finfo(jnp.float64).max))
        return jnp.where(mask, jnp.exp(safe_uv), y)
    raise ValueError("CRRA inverse utility only implemented for theta=0 or theta=1")


def _transition_next_state_jax(state, trade_x, shocks, cfg, sminv, smaxv):
    next_state = jnp.zeros_like(state, dtype=jnp.float64)
    growth_r = jnp.exp(cfg.beta0 + cfg.beta1 * jnp.log(state[:, 0]) + shocks[:, 0])
    growth_p = jnp.exp(cfg.alpha0 + cfg.alpha1 * jnp.log(state[:, 1]) + cfg.alpha2 * jnp.log(state[:, 0]) + shocks[:, 1])
    next_state = next_state.at[:, 0].set(jnp.clip(growth_r, sminv[0], smaxv[0]))
    next_state = next_state.at[:, 1].set(jnp.clip(growth_p, sminv[1], smaxv[1]))
    next_state = next_state.at[:, 2].set(state[:, 2] + trade_x)

    buy_price = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
    sell_mask = trade_x < 0
    buy_price = jnp.where(sell_mask, (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme, buy_price)
    sell_price_now = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    sell_price_next = (1.0 - cfg.tcs) * next_state[:, 1] + (1.0 - cfg.fds) * cfg.fme

    equity_now = state[:, 3] - sell_price_now * state[:, 2]
    base_rate = jnp.full(state.shape[0], cfg.rp, dtype=jnp.float64)
    low_land = next_state[:, 2] < 1.0

    cash_after_trade = equity_now - buy_price * trade_x
    rate_low_land = jnp.where(cash_after_trade < 0, cfg.rn, base_rate)
    wealth_low_land = (1.0 + rate_low_land) * cash_after_trade

    cash_after_cost = equity_now - buy_price * trade_x - cfg.cost * next_state[:, 2]
    rate_high_land = jnp.where(cash_after_cost < 0, cfg.rn, base_rate)
    operating_assets = (1.0 + rate_high_land) * cash_after_cost + next_state[:, 0] * next_state[:, 2]
    wealth_high_land = operating_assets + sell_price_next * next_state[:, 2]

    next_wealth = jnp.where(low_land, wealth_low_land, wealth_high_land)
    return next_state.at[:, 3].set(next_wealth)


def _evaluate_candidate_trade_jax(state, coeffs, t, trade_x, cfg, arrays, shock_nodes, shock_weights):
    state_j = jnp.asarray(state, dtype=jnp.float64)
    trade_x_j = jnp.asarray(trade_x, dtype=jnp.float64)
    shock_nodes_j = jnp.asarray(shock_nodes, dtype=jnp.float64)
    shock_weights_j = jnp.asarray(shock_weights, dtype=jnp.float64)
    sminv_j = jnp.asarray(arrays["sminv"], dtype=jnp.float64)
    smaxv_j = jnp.asarray(arrays["smaxv"], dtype=jnp.float64)
    smin_j = jnp.asarray(arrays["smin"], dtype=jnp.float64)
    smax_j = jnp.asarray(arrays["smax"], dtype=jnp.float64)
    use_crra = bool(getattr(cfg, "CRRA", True))
    coeffs_j = None if coeffs is None else jnp.asarray(coeffs, dtype=jnp.float64)

    def value_for_shock(shock):
        shocks = jnp.broadcast_to(shock, (state_j.shape[0], shock.shape[0]))
        next_state = _transition_next_state_jax(state_j, trade_x_j, shocks, cfg, sminv_j, smaxv_j)
        g4 = next_state[:, 3]
        current_val = jnp.zeros(state_j.shape[0], dtype=jnp.float64)
        low_land = next_state[:, 2] < 1.0
        low_val = _model_utility_jax(((1.0 + jnp.where(g4 < 0, cfg.rn, cfg.rp)) ** (cfg.T - t)) * g4, cfg.b, cfg.theta, use_crra)
        current_val = jnp.where(low_land, low_val, current_val)

        high_land = jnp.logical_not(low_land)
        positive = g4 > 0
        pos_mask = jnp.logical_and(high_land, positive)
        nonpos_mask = jnp.logical_and(high_land, jnp.logical_not(positive))

        if coeffs_j is None:
            pos_val = _model_utility_jax(g4, cfg.b, cfg.theta, use_crra)
            nonpos_val = _model_utility_jax(((1.0 + cfg.rn) ** (cfg.T - t)) * g4, cfg.b, cfg.theta, use_crra)
            current_val = jnp.where(pos_mask, pos_val, current_val)
            current_val = jnp.where(nonpos_mask, nonpos_val, current_val)
        else:
            g4_cap = jnp.minimum(g4, smaxv_j[3])
            residual = jnp.maximum(g4 - smaxv_j[3], 0.0)
            g_eval = next_state.at[:, 3].set(g4_cap)
            phi = [
                _linear_spline_basis_jax(int(arrays["n"][d]), float(smin_j[d]), float(smax_j[d]), g_eval[:, d])
                for d in range(len(arrays["n"]))
            ]
            v_interp = _tensor_basis_interpolate_jax(phi, coeffs_j).reshape(-1)
            wealth_from_interp = _model_inverse_utility_jax(v_interp, cfg.b, cfg.theta, use_crra)
            wealth_adjusted = _model_utility_jax(
                wealth_from_interp + ((1.0 + cfg.rp) ** (cfg.T - t)) * residual,
                cfg.b,
                cfg.theta,
                use_crra,
            )
            v_final = jnp.where(g4 > smaxv_j[3], wealth_adjusted, v_interp)
            pos_val = v_final
            nonpos_val = _model_utility_jax(((1.0 + cfg.rn) ** (cfg.T - t)) * g4, cfg.b, cfg.theta, use_crra)
            current_val = jnp.where(pos_mask, pos_val, current_val)
            current_val = jnp.where(nonpos_mask, nonpos_val, current_val)

        return current_val

    values_by_shock = jax.vmap(value_for_shock)(shock_nodes_j)
    return jnp.sum(values_by_shock * shock_weights_j[:, None], axis=0)


def _maximize_value_v1_jax(state, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    x_low, x_high = feasible_trade_bounds(state, cfg, sminv, smaxv)
    nn = state.shape[0]
    x_candidates = np.zeros((cfg.q + 2, nn), dtype=float)
    v_candidates = np.zeros((cfg.q + 2, nn), dtype=float)
    gap = (x_high - x_low) / (cfg.q - 1)

    arrays = {"n": n, "sminv": sminv, "smaxv": smaxv, "smin": smin, "smax": smax}
    for qi in range(cfg.q + 2):
        if qi == 0:
            x_q = np.zeros(nn, dtype=float)
        elif qi == 1:
            x_q = -state[:, 2]
        else:
            x_q = x_low + gap * (qi - 2)
        x_candidates[qi, :] = x_q
        v_candidates[qi, :] = np.asarray(
            _evaluate_candidate_trade_jax(state, coeffs, t, x_q, cfg, arrays, shock_nodes, shock_weights)
        )

    argmax_idx = np.argmax(v_candidates, axis=0)
    j = np.arange(state.shape[0])
    return x_candidates[argmax_idx, j], v_candidates[argmax_idx, j]


def _maximize_value_v2_jax(state, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights):
    x_low, x_high = feasible_trade_bounds(state, cfg, sminv, smaxv)
    nn = state.shape[0]
    x_coarse = np.zeros((cfg.q, nn), dtype=float)
    v_coarse = np.zeros((cfg.q, nn), dtype=float)
    gap = (x_high - x_low) / (cfg.q - 1)

    arrays = {"n": n, "sminv": sminv, "smaxv": smaxv, "smin": smin, "smax": smax}
    for qi in range(cfg.q):
        x_q = x_low + gap * qi
        x_coarse[qi, :] = x_q
        v_coarse[qi, :] = np.asarray(
            _evaluate_candidate_trade_jax(state, coeffs, t, x_q, cfg, arrays, shock_nodes, shock_weights)
        )

    argmax_coarse = np.argmax(v_coarse, axis=0)
    j = np.arange(state.shape[0])
    x_star_coarse = x_coarse[argmax_coarse, j]

    refine_span = 1600.0 / (cfg.q - 1)
    x_refine_low = np.maximum(x_low, x_star_coarse - refine_span)
    x_refine_high = np.minimum(x_high, x_star_coarse + refine_span)

    x_fine = np.zeros((cfg.qf + 2, nn), dtype=float)
    v_fine = np.zeros((cfg.qf + 2, nn), dtype=float)
    gap_fine = (x_refine_high - x_refine_low) / (cfg.qf - 1)
    for qi in range(cfg.qf + 2):
        if qi == 0:
            x_q = np.zeros(nn, dtype=float)
        elif qi == 1:
            x_q = -state[:, 2]
        else:
            x_q = x_refine_low + gap_fine * (qi - 2)
        x_fine[qi, :] = x_q
        v_fine[qi, :] = np.asarray(
            _evaluate_candidate_trade_jax(state, coeffs, t, x_q, cfg, arrays, shock_nodes, shock_weights)
        )

    argmax_fine = np.argmax(v_fine, axis=0)
    return x_fine[argmax_fine, j], v_fine[argmax_fine, j]


def solve_model_coefficients(cfg, use_jax=False):
    arrays = build_model_arrays(cfg)
    n = arrays["n"]
    m = arrays["m"]
    ee = arrays["Ee"]
    var_cov = arrays["VarCov"]
    sminv = arrays["sminv"]
    smaxv = arrays["smaxv"]
    smin = arrays["smin"]
    smax = arrays["smax"]

    shock_nodes, shock_weights = gauss_hermite_multinormal(m, ee, var_cov)
    state_nodes = [uniform_nodes(int(n[d]), float(smin[d]), float(smax[d])) for d in range(len(n))]
    inverse_basis = [
        np.linalg.inv(linear_spline_basis(int(n[d]), float(smin[d]), float(smax[d]), state_nodes[d]))
        for d in range(len(n))
    ]
    state_grid = cartesian_grid(state_nodes)

    n_points = state_grid.shape[0]
    coeffs_over_time = np.zeros((n_points, cfg.T), dtype=float)
    active_idx = np.where(state_grid[:, 3] > 1.0)[0]
    active_states = state_grid[active_idx]
    coeffs = None

    trade_x = -active_states[:, 2].copy()
    value_maximizer = get_value_maximizer(choose_value_maximizer(cfg))

    if use_jax and jax is None:
        raise ImportError("JAX is not installed. Install jax/jaxlib or set use_jax=False.")

    for t in range(cfg.T, cfg.t1 - 1, -1):
        if use_jax:
            if choose_value_maximizer(cfg) == "vmaxh1":
                trade_x, value = _maximize_value_v1_jax(
                    active_states, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
                )
            else:
                trade_x, value = _maximize_value_v2_jax(
                    active_states, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
                )
        else:
            trade_x, value = value_maximizer(
                active_states, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
            )
        value_full = np.full(
            n_points,
            model_utility(0.0, cfg.b, cfg.theta, bool(getattr(cfg, "CRRA", True))),
            dtype=float,
        )
        value_full[active_idx] = value
        coeffs = apply_inverse_basis_chain(inverse_basis, value_full.reshape(-1, 1)).reshape(-1)
        coeffs_over_time[:, t - 1] = coeffs
    return coeffs_over_time


def compute_policy_grid_local(copt, cfg, t=0, Lt=600.0, Rt=365.0, wealth_levels=None):
    arrays = build_model_arrays(cfg)
    n = arrays["n"]
    shock_nodes, shock_weights = gauss_hermite_multinormal(arrays["m"], arrays["Ee"], arrays["VarCov"])
    sminv = arrays["sminv"]
    smaxv = arrays["smaxv"]
    smin = arrays["smin"]
    smax = arrays["smax"]

    p_vals = np.arange(cfg.sminv[1], cfg.smaxv[1] + 5.0, 5.0, dtype=float)
    if wealth_levels is None:
        w_vals = np.arange(400_000.0, 1_600_000.0 + 150_000.0, 150_000.0, dtype=float)
    else:
        w_vals = np.asarray(wealth_levels, dtype=float)

    coeffs = None if t == cfg.T else copt[:, t]
    value_maximizer = get_value_maximizer(choose_value_maximizer(cfg))
    x_policy = np.zeros((p_vals.size, w_vals.size), dtype=float)

    for wi, wealth_t in enumerate(w_vals):
        state = np.zeros((p_vals.size, len(n)), dtype=float)
        state[:, 0] = Rt
        state[:, 1] = p_vals
        state[:, 2] = Lt
        state[:, 3] = wealth_t
        x_opt, _ = value_maximizer(
            state, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
        )
        x_policy[:, wi] = x_opt

    return {"P": p_vals, "W": w_vals, "XP": x_policy}
