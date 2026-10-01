import utils as utils
import numerical_tools as numtools
import numpy as np
from numba import njit

class PortfolioChoiceModel:

    def __init__(self, config=None, farmland_gross_returns=None, sp500_gross_returns=None):
        self.config = config or utils.PortfolioConfig()
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
            default_farmland, default_sp500 = utils.load_historical_gross_returns(
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

        # Precompute action return matrix
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
        flow_utility = numtools.utility(self.wealth_grid, self.theta, self.b, self.CRRA)
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
        flow_utility = numtools.utility(self.wealth_grid, self.theta, self.b, self.CRRA)

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

    eta = float(getattr(cfg, "cost_scale_eta", 1.0))
    lref = max(float(getattr(cfg, "cost_scale_ref", 600.0)), 1e-12)
    operating_cost = cfg.cost * (lref ** (1.0 - eta)) * np.maximum(next_state[:, 2], 0.0) ** eta
    cash_after_cost = equity_now - buy_price * trade_x - operating_cost
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
    backend = utils._value_backend(cfg)

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
            float(getattr(cfg, "cost_scale_eta", 1.0)),
            float(getattr(cfg, "cost_scale_ref", 600.0)),
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
            current_val[low_land] = numtools.model_utility(
                ((1.0 + rate) ** (cfg.T - t)) * wealth,
                cfg.b,
                cfg.theta,
                use_crra,
            )

        high_land = ~low_land
        if np.any(high_land):
            idx = np.where(high_land)[0]
            if coeffs is None or coeffs.size == 0:
                current_val[idx] = numtools.model_utility(g4[idx], cfg.b, cfg.theta, use_crra)
            else:
                idx_pos = idx[g4[idx] > 0]
                if idx_pos.size > 0:
                    g4_cap = np.minimum(g4[idx_pos], smaxv[3])
                    residual = g4[idx_pos] - g4_cap
                    g_eval = g[idx_pos].copy()
                    g_eval[:, 3] = g4_cap
                    phi = [
                        numtools.linear_spline_basis(int(n[d]), float(smin[d]), float(smax[d]), g_eval[:, d])
                        for d in range(len(n))
                    ]
                    v_interp = numtools.tensor_basis_interpolate(phi, coeffs.reshape(-1, 1)).reshape(-1)
                    wealth_from_interp = numtools.model_inverse_utility(v_interp, cfg.b, cfg.theta, use_crra)
                    over_cap = g4[idx_pos] > smaxv[3]
                    if np.any(over_cap):
                        wealth_from_interp[over_cap] = numtools.model_utility(
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
                    current_val[idx_nonpos] = numtools.model_utility(
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
    arrays = utils.build_model_arrays(cfg)
    n = arrays["n"]
    m = arrays["m"]
    ee = arrays["Ee"]
    var_cov = arrays["VarCov"]
    sminv = arrays["sminv"]
    smaxv = arrays["smaxv"]
    smin = arrays["smin"]
    smax = arrays["smax"]

    shock_nodes, shock_weights = numtools.gauss_hermite_multinormal(m, ee, var_cov)
    state_nodes = [numtools.uniform_nodes(int(n[d]), float(smin[d]), float(smax[d])) for d in range(len(n))]
    inverse_basis = [
        np.linalg.inv(numtools.linear_spline_basis(int(n[d]), float(smin[d]), float(smax[d]), state_nodes[d]))
        for d in range(len(n))
    ]
    state_grid = numtools.cartesian_grid(state_nodes)

    n_points = state_grid.shape[0]
    coeffs_over_time = np.zeros((n_points, cfg.T), dtype=float)
    active_idx = np.where(state_grid[:, 3] > 1.0)[0]
    active_states = state_grid[active_idx]
    coeffs = None

    trade_x = -active_states[:, 2].copy()
    value_maximizer = get_value_maximizer(utils.choose_value_maximizer(cfg))
    for t in range(cfg.T, cfg.t1 - 1, -1):
        trade_x, value = value_maximizer(
            active_states, coeffs, t, cfg, n, sminv, smaxv, smin, smax, shock_nodes, shock_weights
        )
        value_full = np.full(
            n_points,
            numtools.model_utility(0.0, cfg.b, cfg.theta, bool(getattr(cfg, "CRRA", True))),
            dtype=float,
        )
        value_full[active_idx] = value
        coeffs = numtools.apply_inverse_basis_chain(inverse_basis, value_full.reshape(-1, 1)).reshape(-1)
        coeffs_over_time[:, t - 1] = coeffs
    return coeffs_over_time


def compute_policy_grid_local(copt, cfg, t=0, Lt=600.0, Rt=365.0, wealth_levels=None):
    arrays = utils.build_model_arrays(cfg)
    n = arrays["n"]
    shock_nodes, shock_weights = numtools.gauss_hermite_multinormal(arrays["m"], arrays["Ee"], arrays["VarCov"])
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
    value_maximizer = get_value_maximizer(utils.choose_value_maximizer(cfg))
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
    cost_scale_eta,
    cost_scale_ref,
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
                vk = numtools._numba_model_utility_scalar(((1.0 + rate_g4) ** (T - t)) * g4, b, theta, crra)
                acc += shock_weights[k] * vk
                continue

            sell_price_next = (1.0 - tcs) * p_next + (1.0 - fds) * fme
            lpost = land_post if land_post > 0.0 else 0.0
            operating_cost = cost * (cost_scale_ref ** (1.0 - cost_scale_eta)) * (lpost ** cost_scale_eta)
            cash_after_cost = equity_now - buy_price * x - operating_cost
            rate_high = rp if cash_after_cost >= 0.0 else rn
            g4 = (1.0 + rate_high) * cash_after_cost + r_next * land_post + sell_price_next * land_post

            if not has_coeffs:
                vk = numtools._numba_model_utility_scalar(g4, b, theta, crra)
            elif g4 > 0.0:
                g4_cap = g4 if g4 <= smaxv[3] else smaxv[3]
                residual = g4 - g4_cap
                v_interp = numtools._numba_interp4_linear(coeffs, n, smin, smax, r_next, p_next, land_post, g4_cap)
                if g4 > smaxv[3]:
                    wealth_interp = numtools._numba_model_inverse_utility_scalar(v_interp, b, theta, crra)
                    vk = numtools._numba_model_utility_scalar(
                        wealth_interp + ((1.0 + rp) ** (T - t)) * residual,
                        b,
                        theta,
                        crra,
                    )
                else:
                    vk = v_interp
            else:
                vk = numtools._numba_model_utility_scalar(rn_factor * g4, b, theta, crra)

            acc += shock_weights[k] * vk

        value[i] = acc

    return value


