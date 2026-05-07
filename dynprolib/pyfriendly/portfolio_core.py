from dataclasses import dataclass
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numba import njit


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

def debt_to_farm_asset_ratio(P, L, W, tcs=0.06, fds=0.07, fme=300.0):
    pps = (1.0 - tcs) * P + (1.0 - fds) * fme
    lafa = (W / (pps * L)) - 1.0
    return max(0.0, -lafa)
    

def resulting_dfa_after_trade(P, L, W, x, xm=0.0, tcs=0.06, tcb=0.01, fds=0.07, fme=300.0):
    """Compute resulting DFA after farmland trade and mutual-fund investment."""
    pps = (1.0 - tcs) * P + (1.0 - fds) * fme
    ppps = (1.0 + tcb) * P + fme
    if x < 0:
        ppps = pps
    L_post = L + x
    if L_post <= 0.0:
        return 0.0
    
    lafa12f = (W - xm - pps * L - ppps * x) / (pps * L_post)
    return max(0.0, -lafa12f)


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

    def __init__(self, config=None, farmland_gross_returns=None, sp500_gross_returns=None):
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

        # Precompute action return matrix for fast Bellman updates.
        self._action_returns = (
            self.actions[:, 0][:, None] * self.risk_free_gross
            + self.actions[:, 1][:, None] * self.farmland_gross_returns[None, :]
            + self.actions[:, 2][:, None] * self.sp500_gross_returns[None, :]
        )

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
    def value_iteration(self, tol=1e-6, max_iter=2_000, return_history=False):
        V = np.zeros(self.state_size)
        history = []

        for it in range(max_iter):
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
    backend = _value_backend(cfg)

    if backend == "numba" and trade_x is not None:
        coeffs_arr = np.empty(0, dtype=np.float64)
        has_coeffs = False
        if coeffs is not None:
            coeffs_arr = np.ascontiguousarray(np.asarray(coeffs, dtype=np.float64).reshape(-1))
            has_coeffs = coeffs_arr.size > 0
        return _numba_evaluate_candidate_trade_no_mutual(
            np.ascontiguousarray(state, dtype=np.float64),
            np.ascontiguousarray(trade_x, dtype=np.float64),
            coeffs_arr,
            has_coeffs,
            int(t),
            int(cfg.T),
            float(cfg.b),
            float(cfg.theta),
            use_crra,
            float(cfg.beta0),
            float(cfg.beta1),
            float(cfg.alpha0),
            float(cfg.alpha1),
            float(cfg.alpha2),
            float(cfg.cost),
            float(cfg.rp),
            float(cfg.rn),
            float(cfg.tcs),
            float(cfg.tcb),
            float(cfg.fme),
            float(cfg.fds),
            np.ascontiguousarray(n, dtype=np.int64),
            np.ascontiguousarray(smin, dtype=np.float64),
            np.ascontiguousarray(smax, dtype=np.float64),
            np.ascontiguousarray(sminv, dtype=np.float64),
            np.ascontiguousarray(smaxv, dtype=np.float64),
            np.ascontiguousarray(shock_nodes, dtype=np.float64),
            np.ascontiguousarray(shock_weights, dtype=np.float64),
        )

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


def solve_model_coefficients(cfg):
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
    for t in range(cfg.T, cfg.t1 - 1, -1):
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






########################################################################################################################
########################################################################################################################    
#for the model with mutual funds
########################################################################################################################
########################################################################################################################

@dataclass
class MutualFundConfig(PortfolioConfig):
    theta: float = 1.0
    gamma0: float = 0.057757
    EgRM: float = 1.073826
    n: tuple[int, int, int, int] = (7, 7, 5, 21)
    q: int = 41
    q2: int = 25
    qf: int = 21
    search_mode: str = "vectorized"
    value_backend: str = "no" #"numba"
    q_batch: int = 4
    q_batch_max_rows: int = 100_000
    q_coarse_step: int = 4
    q_refine_radius: int = 2
    dau: float = 0.7
    q2_batch_max_rows: int = 100_000
    q2_batch: int = 4
    q2_coarse_step: int = 4
    q2_refine_radius: int = 2
    q2_vectorized_max_rows: int = 100_000
    q2_vectorized_max_rows_with_coeffs: int = 100_000
    two_stage_fine_max_rows: int = 100_000
    two_stage_fine_state_batch: int = 0
    m: tuple[int, int, int] = (3, 3, 3)
    Ee: tuple[float, float, float] = (0.0, 0.0, 0.0)
    var_cov: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (0.030186, 0.0, 0.0),
        (0.0, 0.017292, 0.0),
        (0.0, 0.0, 0.026941),
    )


def mutual_bounds_x(state, cfg):
    state = np.asarray(state, dtype=float)
    x_lower = np.full(state.shape[0], cfg.sminv[2], dtype=float) - state[:, 2]
    x_upper_land = np.full(state.shape[0], cfg.smaxv[2], dtype=float) - state[:, 2]
    sell_price = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    buy_price = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
    x_upper_debt = np.maximum(
        0.0,
        (state[:, 3] - (1.0 - cfg.dau) * sell_price * state[:, 2]) / (buy_price - cfg.dau * sell_price),
    )
    return x_lower, np.minimum(x_upper_land, x_upper_debt)


def mutual_bounds_xm(state, trade_x, cfg):
    state = np.asarray(state, dtype=float)
    trade_x = np.asarray(trade_x, dtype=float)
    if np.isscalar(trade_x):
        trade_x = np.full(state.shape[0], float(trade_x), dtype=float)
    low_land = state[:, 2] + trade_x < 1.0

    # Elementwise handling: rows that sell down below 1 acre should have
    # mutual bounds fixed at current net wealth, regardless of other rows.
    xml = np.zeros(state.shape[0], dtype=float)
    xmu = np.zeros(state.shape[0], dtype=float)
    if np.any(low_land):
        xml[low_land] = state[low_land, 3]
        xmu[low_land] = state[low_land, 3]

    active = ~low_land
    if np.any(active):
        s2bs = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
        sell_mask = trade_x < 0
        s2bs[sell_mask] = (1.0 - cfg.tcs) * state[sell_mask, 1] + (1.0 - cfg.fds) * cfg.fme
        s2v = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
        at = state[:, 3] - s2v * state[:, 2]
        xmu[active] = np.maximum(
            0.0,
            at[active] - s2bs[active] * trade_x[active] + cfg.dau * s2v[active] * (state[active, 2] + trade_x[active]),
        )
    return xml, xmu


def mutual_transition_next_state(state, trade_x, mutual_x, shocks, cfg, sminv, smaxv):
    state = np.asarray(state, dtype=float)
    trade_x = np.asarray(trade_x, dtype=float)
    mutual_x = np.asarray(mutual_x, dtype=float)
    shocks = np.asarray(shocks, dtype=float)
    next_state = np.zeros_like(state, dtype=float)

    growth_r = np.exp(cfg.beta0 + cfg.beta1 * np.log(state[:, 0]) + shocks[:, 0])
    growth_p = np.exp(cfg.alpha0 + cfg.alpha1 * np.log(state[:, 1]) + cfg.alpha2 * np.log(state[:, 0]) + shocks[:, 1])
    growth_m = np.exp(cfg.gamma0 + shocks[:, 2])

    next_state[:, 0] = np.clip(growth_r, sminv[0], smaxv[0])
    next_state[:, 1] = np.clip(growth_p, sminv[1], smaxv[1])
    next_state[:, 2] = state[:, 2] + trade_x

    sell_price_now = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    sell_price_next = (1.0 - cfg.tcs) * next_state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    buy_price = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
    sell_mask = trade_x < 0
    buy_price[sell_mask] = (1.0 - cfg.tcs) * state[sell_mask, 1] + (1.0 - cfg.fds) * cfg.fme

    equity_now = state[:, 3] - sell_price_now * state[:, 2]
    low_land = next_state[:, 2] < 1.0

    cash_after_cost = equity_now - buy_price * trade_x - cfg.cost * next_state[:, 2] - mutual_x
    rate_high_land = np.full(state.shape[0], cfg.rp, dtype=float)
    rate_high_land[cash_after_cost < 0] = cfg.rn
    wealth_high_land = (
        (1.0 + rate_high_land) * cash_after_cost
        + next_state[:, 0] * next_state[:, 2]
        + sell_price_next * next_state[:, 2]
        + growth_m * mutual_x
    )

    # Elementwise low-land transition to match MATLAB gstate.m behavior:
    # if L_{t+1}<1, wealth follows EgRM * W_t.
    next_state[:, 3] = wealth_high_land
    if np.any(low_land):
        next_state[low_land, 3] = cfg.EgRM * state[low_land, 3]
    return next_state


def _search_mode(cfg):
    return str(getattr(cfg, 'search_mode', 'matlab')).lower()


def _value_backend(cfg):
    return str(getattr(cfg, 'value_backend', 'numba')).lower()


from numba import njit


@njit(cache=True)
def _numba_exit_value_scalar(w, t, T, EgRM, gamma0, b, theta, crra):
    if crra and theta == 1.0:
        ww = w if w > 1e-12 else 1e-12
        return np.log(ww) + gamma0 * (T - t + 1)
    return _numba_model_utility_scalar((EgRM ** (T - t + 1)) * w, b, theta, crra)


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
def _numba_mutual_value_function_no_coeffs(
    state,
    trade_x,
    mutual_x,
    t,
    T,
    EgRM,
    beta0,
    beta1,
    alpha0,
    alpha1,
    alpha2,
    gamma0,
    b,
    theta,
    crra,
    tcs,
    tcb,
    fds,
    fme,
    rp,
    rn,
    cost,
    shock_nodes,
    shock_weights,
):
    n_states = state.shape[0]
    n_shocks = shock_nodes.shape[0]
    value = np.zeros(n_states, dtype=np.float64)
    cont_factor = EgRM ** (T - t + 1)

    for i in range(n_states):
        L = state[i, 2]
        W = state[i, 3]
        x = trade_x[i]
        xm = mutual_x[i]

        if L + x < 1.0:
            value[i] = _numba_exit_value_scalar(W, t, T, EgRM, gamma0, b, theta, crra)
            continue

        acc = 0.0
        R = state[i, 0]
        P = state[i, 1]
        sell_price_now = (1.0 - tcs) * P + (1.0 - fds) * fme
        buy_price = (1.0 + tcb) * P + fme
        if x < 0.0:
            buy_price = sell_price_now
        equity_now = W - sell_price_now * L
        land_post = L + x
        cash_after_cost = equity_now - buy_price * x - cost * land_post - xm
        rate_high_land = rp if cash_after_cost >= 0.0 else rn

        for k in range(n_shocks):
            growth_r = np.exp(beta0 + beta1 * np.log(R) + shock_nodes[k, 0])
            growth_p = np.exp(alpha0 + alpha1 * np.log(P) + alpha2 * np.log(R) + shock_nodes[k, 1])
            growth_m = np.exp(gamma0 + shock_nodes[k, 2])
            sell_price_next = (1.0 - tcs) * growth_p + (1.0 - fds) * fme

            if land_post < 1.0:
                vk = _numba_model_utility_scalar(cont_factor * W, b, theta, crra)
            else:
                next_W = (1.0 + rate_high_land) * cash_after_cost + growth_r * land_post + sell_price_next * land_post + growth_m * xm
                vk = _numba_model_utility_scalar(next_W, b, theta, crra)
            acc += shock_weights[k] * vk

        value[i] = acc

    return value


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


@njit(cache=True)
def _numba_evaluate_candidate_trade_no_mutual(
    state,
    trade_x,
    coeffs,
    has_coeffs,
    t,
    T,
    b,
    theta,
    crra,
    beta0,
    beta1,
    alpha0,
    alpha1,
    alpha2,
    cost,
    rp,
    rn,
    tcs,
    tcb,
    fme,
    fds,
    n,
    smin,
    smax,
    sminv,
    smaxv,
    shock_nodes,
    shock_weights,
):
    n_states = state.shape[0]
    n_shocks = shock_nodes.shape[0]
    value = np.zeros(n_states, dtype=np.float64)
    rn_factor = (1.0 + rn) ** (T - t)

    for i in range(n_states):
        R = state[i, 0]
        P = state[i, 1]
        L = state[i, 2]
        W = state[i, 3]
        x = trade_x[i]

        sell_price_now = (1.0 - tcs) * P + (1.0 - fds) * fme
        buy_price = (1.0 + tcb) * P + fme
        if x < 0.0:
            buy_price = sell_price_now
        equity_now = W - sell_price_now * L

        acc = 0.0
        for k in range(n_shocks):
            growth_r = np.exp(beta0 + beta1 * np.log(R) + shock_nodes[k, 0])
            growth_p = np.exp(alpha0 + alpha1 * np.log(P) + alpha2 * np.log(R) + shock_nodes[k, 1])

            r_next = growth_r
            if r_next < sminv[0]:
                r_next = sminv[0]
            elif r_next > smaxv[0]:
                r_next = smaxv[0]

            p_next = growth_p
            if p_next < sminv[1]:
                p_next = sminv[1]
            elif p_next > smaxv[1]:
                p_next = smaxv[1]

            land_post = L + x
            if land_post < 1.0:
                cash_after_trade = equity_now - buy_price * x
                rate_low = rp if cash_after_trade >= 0.0 else rn
                g4 = (1.0 + rate_low) * cash_after_trade
                rate_g4 = rp if g4 >= 0.0 else rn
                vk = _numba_model_utility_scalar(((1.0 + rate_g4) ** (T - t)) * g4, b, theta, crra)
                acc += shock_weights[k] * vk
                continue

            sell_price_next = (1.0 - tcs) * p_next + (1.0 - fds) * fme
            cash_after_cost = equity_now - buy_price * x - cost * land_post
            rate_high = rp if cash_after_cost >= 0.0 else rn
            g4 = (1.0 + rate_high) * cash_after_cost + r_next * land_post + sell_price_next * land_post

            if not has_coeffs:
                vk = _numba_model_utility_scalar(g4, b, theta, crra)
            elif g4 > 0.0:
                g4_cap = g4 if g4 <= smaxv[3] else smaxv[3]
                residual = g4 - g4_cap
                v_interp = _numba_interp4_linear(coeffs, n, smin, smax, r_next, p_next, land_post, g4_cap)
                if g4 > smaxv[3]:
                    wealth_interp = _numba_model_inverse_utility_scalar(v_interp, b, theta, crra)
                    vk = _numba_model_utility_scalar(
                        wealth_interp + ((1.0 + rp) ** (T - t)) * residual,
                        b,
                        theta,
                        crra,
                    )
                else:
                    vk = v_interp
            else:
                vk = _numba_model_utility_scalar(rn_factor * g4, b, theta, crra)

            acc += shock_weights[k] * vk

        value[i] = acc

    return value


@njit(cache=True)
def _numba_mutual_value_function_with_coeffs(
    state,
    trade_x,
    mutual_x,
    coeffs,
    t,
    T,
    EgRM,
    beta0,
    beta1,
    alpha0,
    alpha1,
    alpha2,
    gamma0,
    b,
    theta,
    crra,
    tcs,
    tcb,
    fds,
    fme,
    rp,
    rn,
    cost,
    n,
    smin,
    smax,
    sminv,
    smaxv,
    shock_nodes,
    shock_weights,
):
    n_states = state.shape[0]
    n_shocks = shock_nodes.shape[0]
    value = np.zeros(n_states, dtype=np.float64)
    cont_factor = EgRM ** (T - t + 1)
    rn_factor = (1.0 + rn) ** (T - t)

    for i in range(n_states):
        L = state[i, 2]
        W = state[i, 3]
        x = trade_x[i]
        xm = mutual_x[i]
        land_post = L + x

        if land_post < 1.0:
            value[i] = _numba_exit_value_scalar(W, t, T, EgRM, gamma0, b, theta, crra)
            continue

        R = state[i, 0]
        P = state[i, 1]
        sell_price_now = (1.0 - tcs) * P + (1.0 - fds) * fme
        buy_price = (1.0 + tcb) * P + fme
        if x < 0.0:
            buy_price = sell_price_now
        equity_now = W - sell_price_now * L
        cash_after_cost = equity_now - buy_price * x - cost * land_post - xm
        rate_high_land = rp if cash_after_cost >= 0.0 else rn

        acc = 0.0
        for k in range(n_shocks):
            growth_r = np.exp(beta0 + beta1 * np.log(R) + shock_nodes[k, 0])
            growth_p = np.exp(alpha0 + alpha1 * np.log(P) + alpha2 * np.log(R) + shock_nodes[k, 1])
            growth_m = np.exp(gamma0 + shock_nodes[k, 2])

            r_next = growth_r
            if r_next < sminv[0]:
                r_next = sminv[0]
            elif r_next > smaxv[0]:
                r_next = smaxv[0]

            p_next = growth_p
            if p_next < sminv[1]:
                p_next = sminv[1]
            elif p_next > smaxv[1]:
                p_next = smaxv[1]

            sell_price_next = (1.0 - tcs) * p_next + (1.0 - fds) * fme
            next_W = (1.0 + rate_high_land) * cash_after_cost + r_next * land_post + sell_price_next * land_post + growth_m * xm

            if next_W > 0.0:
                w_cap = next_W if next_W <= smaxv[3] else smaxv[3]
                v_interp = _numba_interp4_linear(coeffs, n, smin, smax, r_next, p_next, land_post, w_cap)
                if next_W > smaxv[3]:
                    tw = _numba_model_inverse_utility_scalar(v_interp, b, theta, crra)
                    vk = _numba_model_utility_scalar(tw + (growth_m ** (T - t)) * (next_W - w_cap), b, theta, crra)
                else:
                    vk = v_interp
            else:
                vk = _numba_model_utility_scalar(rn_factor * next_W, b, theta, crra)

            acc += shock_weights[k] * vk

        value[i] = acc

    return value


def mutual_value_function(state, coeffs, t, controls, cfg, arrays, shock_nodes, shock_weights):
    state = np.asarray(state, dtype=float)
    controls = np.asarray(controls, dtype=float)
    trade_x = controls[:, 0]
    mutual_x = controls[:, 1]
    use_crra = bool(getattr(cfg, "CRRA", True))
    mode = _search_mode(cfg)
    backend = _value_backend(cfg)

    if backend == 'numba' and mode != 'matlab':
        state64 = np.ascontiguousarray(state, dtype=np.float64)
        trade64 = np.ascontiguousarray(trade_x, dtype=np.float64)
        mutual64 = np.ascontiguousarray(mutual_x, dtype=np.float64)
        shocks64 = np.ascontiguousarray(shock_nodes, dtype=np.float64)
        weights64 = np.ascontiguousarray(shock_weights, dtype=np.float64)

        if coeffs is None or coeffs.size == 0:
            return _numba_mutual_value_function_no_coeffs(
                state64,
                trade64,
                mutual64,
                int(t),
                int(cfg.T),
                float(cfg.EgRM),
                float(cfg.beta0),
                float(cfg.beta1),
                float(cfg.alpha0),
                float(cfg.alpha1),
                float(cfg.alpha2),
                float(cfg.gamma0),
                float(cfg.b),
                float(cfg.theta),
                use_crra,
                float(cfg.tcs),
                float(cfg.tcb),
                float(cfg.fds),
                float(cfg.fme),
                float(cfg.rp),
                float(cfg.rn),
                float(cfg.cost),
                shocks64,
                weights64,
            )

        coeffs64 = np.ascontiguousarray(np.asarray(coeffs, dtype=np.float64).reshape(-1))
        return _numba_mutual_value_function_with_coeffs(
            state64,
            trade64,
            mutual64,
            coeffs64,
            int(t),
            int(cfg.T),
            float(cfg.EgRM),
            float(cfg.beta0),
            float(cfg.beta1),
            float(cfg.alpha0),
            float(cfg.alpha1),
            float(cfg.alpha2),
            float(cfg.gamma0),
            float(cfg.b),
            float(cfg.theta),
            use_crra,
            float(cfg.tcs),
            float(cfg.tcb),
            float(cfg.fds),
            float(cfg.fme),
            float(cfg.rp),
            float(cfg.rn),
            float(cfg.cost),
            np.ascontiguousarray(arrays['n'], dtype=np.int64),
            np.ascontiguousarray(arrays['smin'], dtype=np.float64),
            np.ascontiguousarray(arrays['smax'], dtype=np.float64),
            np.ascontiguousarray(arrays['sminv'], dtype=np.float64),
            np.ascontiguousarray(arrays['smaxv'], dtype=np.float64),
            shocks64,
            weights64,
        )

    if mode == 'matlab':
        value = np.zeros(state.shape[0], dtype=float)
        for k in range(shock_nodes.shape[0]):
            shocks = np.tile(shock_nodes[k], (state.shape[0], 1))
            next_state = mutual_transition_next_state(state, trade_x, mutual_x, shocks, cfg, arrays['sminv'], arrays['smaxv'])
            v_k = np.zeros(state.shape[0], dtype=float)
            low_land = next_state[:, 2] < 1.0

            if np.all(low_land):
                if use_crra and cfg.theta == 1.0:
                    v_k[:] = np.log(np.maximum(state[:, 3], 1e-12)) + cfg.gamma0 * (cfg.T - t + 1)
                else:
                    continuation_wealth = (cfg.EgRM ** (cfg.T - t + 1)) * state[:, 3]
                    v_k[:] = model_utility(continuation_wealth, cfg.b, cfg.theta, use_crra)
            else:
                gf4 = next_state[:, 3]
                if coeffs is None or coeffs.size == 0:
                    v_k[:] = model_utility(gf4, cfg.b, cfg.theta, use_crra)
                else:
                    nbri = np.where(gf4 > 0)[0]
                    if nbri.size > 0:
                        gm4 = np.minimum(next_state[nbri, 3], arrays['smaxv'][3])
                        gd = next_state[nbri, 3] - gm4
                        g_eval = next_state[nbri].copy()
                        g_eval[:, 3] = gm4
                        phi = [
                            linear_spline_basis(int(arrays['n'][d]), float(arrays['smin'][d]), float(arrays['smax'][d]), g_eval[:, d])
                            for d in range(len(arrays['n']))
                        ]
                        v_interp = tensor_basis_interpolate(phi, coeffs.reshape(-1, 1)).reshape(-1)
                        tw = model_inverse_utility(v_interp, cfg.b, cfg.theta, use_crra)
                        v_nbri = v_interp.copy()
                        over_cap = next_state[nbri, 3] > arrays['smaxv'][3]
                        if np.any(over_cap):
                            gRM_k = np.exp(cfg.gamma0 + shock_nodes[k, 2])
                            v_nbri[over_cap] = model_utility(
                                tw[over_cap] + (gRM_k ** (cfg.T - t)) * gd[over_cap],
                                cfg.b,
                                cfg.theta,
                                use_crra,
                            )
                        v_k[nbri] = v_nbri
                    neg = np.where(gf4 <= 0)[0]
                    if neg.size > 0:
                        v_k[neg] = model_utility(
                            ((1.0 + cfg.rn) ** (cfg.T - t)) * gf4[neg],
                            cfg.b,
                            cfg.theta,
                            use_crra,
                        )

            value += v_k * shock_weights[k]
        return value

    low_land = state[:, 2] + trade_x < 1.0
    value = np.zeros(state.shape[0], dtype=float)
    if np.any(low_land):
        if use_crra and cfg.theta == 1.0:
            value[low_land] = np.log(np.maximum(state[low_land, 3], 1e-12)) + cfg.gamma0 * (cfg.T - t + 1)
        else:
            continuation_wealth = (cfg.EgRM ** (cfg.T - t + 1)) * state[low_land, 3]
            value[low_land] = model_utility(continuation_wealth, cfg.b, cfg.theta, use_crra)

    active = np.where(~low_land)[0]
    if active.size == 0:
        return value

    n_active = active.size
    n_shocks = shock_nodes.shape[0]
    state_active = state[active]
    trade_active = trade_x[active]
    mutual_active = mutual_x[active]
    state_rep = np.repeat(state_active, n_shocks, axis=0)
    trade_rep = np.repeat(trade_active, n_shocks)
    mutual_rep = np.repeat(mutual_active, n_shocks)
    shocks_rep = np.tile(shock_nodes, (n_active, 1))

    growth_r = np.exp(cfg.beta0 + cfg.beta1 * np.log(state_rep[:, 0]) + shocks_rep[:, 0])
    growth_p = np.exp(cfg.alpha0 + cfg.alpha1 * np.log(state_rep[:, 1]) + cfg.alpha2 * np.log(state_rep[:, 0]) + shocks_rep[:, 1])
    growth_m = np.exp(cfg.gamma0 + shocks_rep[:, 2])

    next_state = np.zeros_like(state_rep, dtype=float)
    next_state[:, 0] = np.clip(growth_r, arrays['sminv'][0], arrays['smaxv'][0])
    next_state[:, 1] = np.clip(growth_p, arrays['sminv'][1], arrays['smaxv'][1])
    next_state[:, 2] = state_rep[:, 2] + trade_rep

    sell_price_now = (1.0 - cfg.tcs) * state_rep[:, 1] + (1.0 - cfg.fds) * cfg.fme
    sell_price_next = (1.0 - cfg.tcs) * next_state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    buy_price = (1.0 + cfg.tcb) * state_rep[:, 1] + cfg.fme
    sell_mask = trade_rep < 0
    buy_price[sell_mask] = (1.0 - cfg.tcs) * state_rep[sell_mask, 1] + (1.0 - cfg.fds) * cfg.fme

    equity_now = state_rep[:, 3] - sell_price_now * state_rep[:, 2]
    cash_after_cost = equity_now - buy_price * trade_rep - cfg.cost * next_state[:, 2] - mutual_rep
    rate_high_land = np.full(state_rep.shape[0], cfg.rp, dtype=float)
    rate_high_land[cash_after_cost < 0] = cfg.rn
    next_state[:, 3] = (1.0 + rate_high_land) * cash_after_cost + next_state[:, 0] * next_state[:, 2] + sell_price_next * next_state[:, 2] + growth_m * mutual_rep

    gf4 = next_state[:, 3]
    if coeffs is None or coeffs.size == 0:
        v_flat = model_utility(gf4, cfg.b, cfg.theta, use_crra)
    else:
        v_flat = np.zeros(gf4.shape[0], dtype=float)
        pos = gf4 > 0
        if np.any(pos):
            nbri = np.where(pos)[0]
            gm4 = np.minimum(next_state[nbri, 3], arrays['smaxv'][3])
            gd = next_state[nbri, 3] - gm4
            g_eval = next_state[nbri].copy()
            g_eval[:, 3] = gm4
            phi = [
                linear_spline_basis(int(arrays['n'][d]), float(arrays['smin'][d]), float(arrays['smax'][d]), g_eval[:, d])
                for d in range(len(arrays['n']))
            ]
            v_interp = tensor_basis_interpolate(phi, coeffs.reshape(-1, 1)).reshape(-1)
            tw = model_inverse_utility(v_interp, cfg.b, cfg.theta, use_crra)
            v_pos = v_interp.copy()
            over_cap = next_state[nbri, 3] > arrays['smaxv'][3]
            if np.any(over_cap):
                v_pos[over_cap] = model_utility(
                    tw[over_cap] + (growth_m[nbri][over_cap] ** (cfg.T - t)) * gd[over_cap],
                    cfg.b,
                    cfg.theta,
                    use_crra,
                )
            v_flat[nbri] = v_pos
        neg = np.where(~pos)[0]
        if neg.size > 0:
            v_flat[neg] = model_utility(
                ((1.0 + cfg.rn) ** (cfg.T - t)) * gf4[neg],
                cfg.b,
                cfg.theta,
                use_crra,
            )

    value[active] = v_flat.reshape(n_active, n_shocks) @ shock_weights
    return value


def mutual_vmaxhm(state, coeffs, t, xqi, cfg, arrays, shock_nodes, shock_weights):
    state = np.asarray(state, dtype=float)
    xqi = np.asarray(xqi, dtype=float)
    xml, xmu = mutual_bounds_xm(state, xqi, cfg)
    nn = state.shape[0]
    mode = _search_mode(cfg)

    if cfg.q2 <= 1:
        controls = np.column_stack([xqi, xml])
        v = mutual_value_function(state, coeffs, t, controls, cfg, arrays, shock_nodes, shock_weights)
        return xml.copy(), v

    gap = (xmu - xml) / (cfg.q2 - 1)

    if mode == 'matlab':
        xmq2 = np.zeros((cfg.q2, nn), dtype=float)
        vxq2 = np.zeros((cfg.q2, nn), dtype=float)
        for qj in range(cfg.q2):
            xmqj = xml + gap * float(qj)
            controls = np.column_stack([xqi, xmqj])
            vxqj = mutual_value_function(state, coeffs, t, controls, cfg, arrays, shock_nodes, shock_weights)
            xmq2[qj, :] = xmqj
            vxq2[qj, :] = vxqj
        ind = np.argmax(vxq2, axis=0)
        j = np.arange(nn)
        xmqi = xmq2[ind, j]
        vqi = vxq2[ind, j]
        return xmqi, vqi

    # Vectorized search over q2 is fast, but can OOM for large nn.
    # Use tighter budgets when continuation interpolation is active (coeffs provided).
    if coeffs is None or (hasattr(coeffs, 'size') and coeffs.size == 0):
        max_rows = int(getattr(cfg, 'q2_vectorized_max_rows', 120_000))
    else:
        max_rows = int(getattr(cfg, 'q2_vectorized_max_rows_with_coeffs', 40_000))

    if nn * int(cfg.q2) <= max_rows:
        xmq = xml[None, :] + gap[None, :] * np.arange(cfg.q2, dtype=float)[:, None]
        state_rep = np.repeat(state[None, :, :], cfg.q2, axis=0).reshape(-1, state.shape[1])
        trade_rep = np.repeat(xqi[None, :], cfg.q2, axis=0).reshape(-1)
        controls_rep = np.column_stack([trade_rep, xmq.reshape(-1)])
        value_rep = mutual_value_function(state_rep, coeffs, t, controls_rep, cfg, arrays, shock_nodes, shock_weights)
        vxq2 = value_rep.reshape(cfg.q2, nn)
        ind = np.argmax(vxq2, axis=0)
        j = np.arange(nn)
        return xmq[ind, j], vxq2[ind, j]

    # Fallback: batch over state dimension to bound memory.
    state_batch = max(1, max_rows // max(1, int(cfg.q2)))
    q_idx = np.arange(cfg.q2, dtype=float)

    best_xm = np.empty(nn, dtype=float)
    best_v = np.empty(nn, dtype=float)

    for s0 in range(0, nn, state_batch):
        s1 = min(s0 + state_batch, nn)
        st = state[s0:s1]
        xq = xqi[s0:s1]
        xml_s = xml[s0:s1]
        gap_s = gap[s0:s1]
        bs = s1 - s0

        xmq = xml_s[None, :] + gap_s[None, :] * q_idx[:, None]
        state_rep = np.repeat(st[None, :, :], cfg.q2, axis=0).reshape(-1, st.shape[1])
        trade_rep = np.repeat(xq[None, :], cfg.q2, axis=0).reshape(-1)
        controls_rep = np.column_stack([trade_rep, xmq.reshape(-1)])
        value_rep = mutual_value_function(state_rep, coeffs, t, controls_rep, cfg, arrays, shock_nodes, shock_weights)

        vxq2 = value_rep.reshape(cfg.q2, bs)
        ind = np.argmax(vxq2, axis=0)
        j = np.arange(bs)
        best_xm[s0:s1] = xmq[ind, j]
        best_v[s0:s1] = vxq2[ind, j]

    return best_xm, best_v


def mutual_vmaxh(state, coeffs, t, cfg, arrays, shock_nodes, shock_weights):
    xl, xu = mutual_bounds_x(state, cfg)
    nn = state.shape[0]
    mode = _search_mode(cfg)
    nq = cfg.q + 2
    xq = np.zeros((nq, nn), dtype=float)

    xq[0, :] = 0.0
    xq[1, :] = -state[:, 2]
    if cfg.q <= 1:
        gap = np.zeros(nn, dtype=float)
    else:
        gap = (xu - xl) / (cfg.q - 1)

    # MATLAB vx.m: for qi >= 3, xqi = xl + gap * (qi - 3)
    if cfg.q > 0:
        q_idx = np.arange(cfg.q, dtype=float)
        xq[2:, :] = xl[None, :] + q_idx[:, None] * gap[None, :]

    def _evaluate_q_subset(q_indices):
        q_indices = np.asarray(q_indices, dtype=int)
        k = q_indices.size
        vx_local = np.zeros((k, nn), dtype=float)
        xm_local = np.zeros((k, nn), dtype=float)

        if mode == 'matlab':
            for i, qi in enumerate(q_indices):
                x_block = xq[qi, :]
                xmqi, vqi = mutual_vmaxhm(state, coeffs, t, x_block, cfg, arrays, shock_nodes, shock_weights)
                xm_local[i, :] = xmqi
                vx_local[i, :] = vqi
            return vx_local, xm_local

        q_batch = max(1, int(getattr(cfg, 'q_batch', 4)))
        max_rows = int(getattr(cfg, 'q_batch_max_rows', 100_000))
        if nn * int(cfg.q2) * q_batch > max_rows:
            q_batch = max(1, max_rows // max(1, nn * int(cfg.q2)))

        for start in range(0, k, q_batch):
            end = min(start + q_batch, k)
            block = end - start
            q_block = q_indices[start:end]
            x_block = xq[q_block, :]

            if block == 1:
                xmqi, vqi = mutual_vmaxhm(
                    state,
                    coeffs,
                    t,
                    x_block[0, :],
                    cfg,
                    arrays,
                    shock_nodes,
                    shock_weights,
                )
                xm_local[start, :] = xmqi
                vx_local[start, :] = vqi
                continue

            state_block = np.repeat(state[None, :, :], block, axis=0).reshape(-1, state.shape[1])
            x_flat = x_block.reshape(-1)
            xm_flat, v_flat = mutual_vmaxhm(
                state_block,
                coeffs,
                t,
                x_flat,
                cfg,
                arrays,
                shock_nodes,
                shock_weights,
            )
            xm_local[start:end, :] = xm_flat.reshape(block, nn)
            vx_local[start:end, :] = v_flat.reshape(block, nn)

        return vx_local, xm_local

    if mode == 'two_stage':
        coarse_step = max(1, int(getattr(cfg, 'q_coarse_step', 4)))
        coarse_idx = np.arange(0, nq, coarse_step, dtype=int)
        if coarse_idx[-1] != nq - 1:
            coarse_idx = np.append(coarse_idx, nq - 1)
        vx_coarse, xm_coarse = _evaluate_q_subset(coarse_idx)
        coarse_best_pos = np.argmax(vx_coarse, axis=0)
        coarse_best_idx = coarse_idx[coarse_best_pos]
        j_idx = np.arange(nn)
        x_coarse = xq[coarse_best_idx, j_idx]
        
        # Initialize with coarse results to avoid discarding them
        best_v = vx_coarse[coarse_best_pos, j_idx].copy()
        best_x = x_coarse.copy()
        best_xm = xm_coarse[coarse_best_pos, j_idx].copy()
        
        xas = 1600.0 / max(1.0, (cfg.q - 1))
        xlf = np.maximum(xl, x_coarse - xas)
        xuf = np.minimum(xu, x_coarse + xas)
        # If qf is not set, use a smaller refinement grid by default.
        fine_q = int(getattr(cfg, 'qf', max(9, cfg.q // 2)))
        fine_q = max(1, min(fine_q, int(cfg.q)))
        fine_nq = fine_q + 2

        # Batch fine search by STATES (not by total nn). The old formula
        # could shrink to batch_size=1 and kill performance.
        max_rows = int(getattr(cfg, 'two_stage_fine_max_rows', getattr(cfg, 'q_batch_max_rows', 100_000)))
        fine_batch = int(getattr(cfg, 'two_stage_fine_state_batch', 0))
        if fine_batch <= 0:
            fine_batch = max(1, max_rows // max(1, fine_nq))

        for batch_start in range(0, nn, fine_batch):
            batch_end = min(batch_start + fine_batch, nn)
            batch_size = batch_end - batch_start

            state_batch = state[batch_start:batch_end]
            state_batch_rep = np.repeat(state_batch[None, :, :], fine_nq, axis=0).reshape(-1, state.shape[1])

            # VECTORIZED: Build all x_candidates at once using broadcasting
            batch_indices = np.arange(batch_start, batch_end)
            x_candidates = np.zeros((batch_size, fine_nq), dtype=float)
            x_candidates[:, 0] = 0.0
            x_candidates[:, 1] = -state_batch[:, 2]
            if fine_q > 1:
                gap_fine = (xuf[batch_indices] - xlf[batch_indices]) / (fine_q - 1)
                idxs = np.arange(fine_nq - 2, dtype=float)
                x_candidates[:, 2:] = xlf[batch_indices, None] + idxs[None, :] * gap_fine[:, None]
            x_batch_ref = x_candidates.reshape(-1)

            xm_batch, v_batch = mutual_vmaxhm(
                state_batch_rep,
                coeffs,
                t,
                x_batch_ref,
                cfg,
                arrays,
                shock_nodes,
                shock_weights,
            )

            # VECTORIZED: Find best value per state using argmax instead of loop
            v_batch_reshaped = v_batch.reshape(batch_size, fine_nq)
            best_idx_per_state = np.argmax(v_batch_reshaped, axis=1)
            best_v_per_state = v_batch_reshaped[np.arange(batch_size), best_idx_per_state]

            # Update only where new values are better
            better_mask = best_v_per_state > best_v[batch_indices]
            if np.any(better_mask):
                update_idx = batch_indices[better_mask]
                local_idx = np.arange(batch_size)[better_mask]
                best_idx_local = best_idx_per_state[local_idx]

                best_v[update_idx] = best_v_per_state[better_mask]
                best_x[update_idx] = x_candidates[local_idx, best_idx_local]
                best_xm[update_idx] = xm_batch.reshape(batch_size, fine_nq)[local_idx, best_idx_local]

        return best_x, best_xm, best_v

    full_idx = np.arange(nq, dtype=int)
    vxq, xmq = _evaluate_q_subset(full_idx)
    ind = np.argmax(vxq, axis=0)
    j = np.arange(nn)
    return xq[full_idx[ind], j], xmq[ind, j], vxq[ind, j]




class MutualFundModel:
    def __init__(self, config=None):
        self.config = config or MutualFundConfig()

    def transition(self, R, P, L, W, x, xm, eps1, eps2, eps3):
        state = np.array([[R, P, L, W]], dtype=float)
        shocks = np.array([[eps1, eps2, eps3]], dtype=float)
        next_state = mutual_transition_next_state(
            state=state,
            trade_x=np.array([x], dtype=float),
            mutual_x=np.array([xm], dtype=float),
            shocks=shocks,
            cfg=self.config,
            sminv=np.array(self.config.sminv, dtype=float),
            smaxv=np.array(self.config.smaxv, dtype=float),
        )
        return tuple(next_state[0])
def solve_mutual_coefficients(cfg):
    arrays = build_model_arrays(cfg)
    n = arrays['n']
    shock_nodes, shock_weights = gauss_hermite_multinormal(arrays['m'], arrays['Ee'], arrays['VarCov'])
    state_nodes = [uniform_nodes(int(n[d]), float(arrays['smin'][d]), float(arrays['smax'][d])) for d in range(len(n))]
    inverse_basis = [
        np.linalg.inv(linear_spline_basis(int(n[d]), float(arrays['smin'][d]), float(arrays['smax'][d]), state_nodes[d]))
        for d in range(len(n))
    ]
    state_grid = cartesian_grid(state_nodes)

    n_points = state_grid.shape[0]
    coeffs_over_time = np.zeros((n_points, cfg.T), dtype=float)
    active_idx = np.where(state_grid[:, 3] > 1.0)[0]
    active_states = state_grid[active_idx]
    coeffs = None
    use_crra = bool(getattr(cfg, "CRRA", True))

    for t in range(cfg.T, cfg.t1 - 1, -1):
        _, _, value = mutual_vmaxh(active_states, coeffs, t, cfg, arrays, shock_nodes, shock_weights)
        value_full = np.full(n_points, model_utility(0.0, cfg.b, cfg.theta, use_crra), dtype=float)
        value_full[active_idx] = value
        coeffs = apply_inverse_basis_chain(inverse_basis, value_full.reshape(-1, 1)).reshape(-1)
        coeffs_over_time[:, t - 1] = coeffs

    return coeffs_over_time


