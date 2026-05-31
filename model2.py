import utils as utils
import numerical_tools as numtools
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass
class MutualFundConfig(utils.PortfolioConfig):
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

    eta = float(getattr(cfg, "cost_scale_eta", 1.0))
    lref = max(float(getattr(cfg, "cost_scale_ref", 600.0)), 1e-12)
    operating_cost = cfg.cost * (lref ** (1.0 - eta)) * np.maximum(next_state[:, 2], 0.0) ** eta
    cash_after_cost = equity_now - buy_price * trade_x - operating_cost - mutual_x
    rate_high_land = np.full(state.shape[0], cfg.rp, dtype=float)
    rate_high_land[cash_after_cost < 0] = cfg.rn
    wealth_high_land = (
        (1.0 + rate_high_land) * cash_after_cost
        + next_state[:, 0] * next_state[:, 2]
        + sell_price_next * next_state[:, 2]
        + growth_m * mutual_x
    )

    next_state[:, 3] = wealth_high_land
    if np.any(low_land):
        next_state[low_land, 3] = cfg.EgRM * state[low_land, 3]
    return next_state


def _search_mode(cfg):
    return str(getattr(cfg, 'search_mode', 'matlab')).lower()


@njit(cache=True)
def _numba_exit_value_scalar(w, t, T, EgRM, gamma0, b, theta, crra):
    if crra and theta == 1.0:
        ww = w if w > 1e-12 else 1e-12
        return np.log(ww) + gamma0 * (T - t + 1)
    return numtools._numba_model_utility_scalar((EgRM ** (T - t + 1)) * w, b, theta, crra)


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
    cost_scale_eta,
    cost_scale_ref,
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
        lpost = land_post if land_post > 0.0 else 0.0
        operating_cost = cost * (cost_scale_ref ** (1.0 - cost_scale_eta)) * (lpost ** cost_scale_eta)
        cash_after_cost = equity_now - buy_price * x - operating_cost - xm
        rate_high_land = rp if cash_after_cost >= 0.0 else rn

        for k in range(n_shocks):
            growth_r = np.exp(beta0 + beta1 * np.log(R) + shock_nodes[k, 0])
            growth_p = np.exp(alpha0 + alpha1 * np.log(P) + alpha2 * np.log(R) + shock_nodes[k, 1])
            growth_m = np.exp(gamma0 + shock_nodes[k, 2])
            sell_price_next = (1.0 - tcs) * growth_p + (1.0 - fds) * fme

            if land_post < 1.0:
                vk = numtools._numba_model_utility_scalar(cont_factor * W, b, theta, crra)
            else:
                next_W = (1.0 + rate_high_land) * cash_after_cost + growth_r * land_post + sell_price_next * land_post + growth_m * xm
                vk = numtools._numba_model_utility_scalar(next_W, b, theta, crra)
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
    cost_scale_eta,
    cost_scale_ref,
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
        lpost = land_post if land_post > 0.0 else 0.0
        operating_cost = cost * (cost_scale_ref ** (1.0 - cost_scale_eta)) * (lpost ** cost_scale_eta)
        cash_after_cost = equity_now - buy_price * x - operating_cost - xm
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
                v_interp = numtools._numba_interp4_linear(coeffs, n, smin, smax, r_next, p_next, land_post, w_cap)
                if next_W > smaxv[3]:
                    tw = numtools._numba_model_inverse_utility_scalar(v_interp, b, theta, crra)
                    vk = numtools._numba_model_utility_scalar(tw + (growth_m ** (T - t)) * (next_W - w_cap), b, theta, crra)
                else:
                    vk = v_interp
            else:
                vk = numtools._numba_model_utility_scalar(rn_factor * next_W, b, theta, crra)

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
    backend = utils._value_backend(cfg)

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
                float(getattr(cfg, "cost_scale_eta", 1.0)),
                float(getattr(cfg, "cost_scale_ref", 600.0)),
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
            float(getattr(cfg, "cost_scale_eta", 1.0)),
            float(getattr(cfg, "cost_scale_ref", 600.0)),
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
                    v_k[:] = numtools.model_utility(continuation_wealth, cfg.b, cfg.theta, use_crra)
            else:
                gf4 = next_state[:, 3]
                if coeffs is None or coeffs.size == 0:
                    v_k[:] = numtools.model_utility(gf4, cfg.b, cfg.theta, use_crra)
                else:
                    nbri = np.where(gf4 > 0)[0]
                    if nbri.size > 0:
                        gm4 = np.minimum(next_state[nbri, 3], arrays['smaxv'][3])
                        gd = next_state[nbri, 3] - gm4
                        g_eval = next_state[nbri].copy()
                        g_eval[:, 3] = gm4
                        phi = [
                            numtools.linear_spline_basis(int(arrays['n'][d]), float(arrays['smin'][d]), float(arrays['smax'][d]), g_eval[:, d])
                            for d in range(len(arrays['n']))
                        ]
                        v_interp = numtools.tensor_basis_interpolate(phi, coeffs.reshape(-1, 1)).reshape(-1)
                        tw = numtools.model_inverse_utility(v_interp, cfg.b, cfg.theta, use_crra)
                        v_nbri = v_interp.copy()
                        over_cap = next_state[nbri, 3] > arrays['smaxv'][3]
                        if np.any(over_cap):
                            gRM_k = np.exp(cfg.gamma0 + shock_nodes[k, 2])
                            v_nbri[over_cap] = numtools.model_utility(
                                tw[over_cap] + (gRM_k ** (cfg.T - t)) * gd[over_cap],
                                cfg.b,
                                cfg.theta,
                                use_crra,
                            )
                        v_k[nbri] = v_nbri
                    neg = np.where(gf4 <= 0)[0]
                    if neg.size > 0:
                        v_k[neg] = numtools.model_utility(
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
            value[low_land] = numtools.model_utility(continuation_wealth, cfg.b, cfg.theta, use_crra)

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
    eta = float(getattr(cfg, "cost_scale_eta", 1.0))
    lref = max(float(getattr(cfg, "cost_scale_ref", 600.0)), 1e-12)
    operating_cost = cfg.cost * (lref ** (1.0 - eta)) * np.maximum(next_state[:, 2], 0.0) ** eta
    cash_after_cost = equity_now - buy_price * trade_rep - operating_cost - mutual_rep
    rate_high_land = np.full(state_rep.shape[0], cfg.rp, dtype=float)
    rate_high_land[cash_after_cost < 0] = cfg.rn
    next_state[:, 3] = (1.0 + rate_high_land) * cash_after_cost + next_state[:, 0] * next_state[:, 2] + sell_price_next * next_state[:, 2] + growth_m * mutual_rep

    gf4 = next_state[:, 3]
    if coeffs is None or coeffs.size == 0:
        v_flat = numtools.model_utility(gf4, cfg.b, cfg.theta, use_crra)
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
                numtools.linear_spline_basis(int(arrays['n'][d]), float(arrays['smin'][d]), float(arrays['smax'][d]), g_eval[:, d])
                for d in range(len(arrays['n']))
            ]
            v_interp = numtools.tensor_basis_interpolate(phi, coeffs.reshape(-1, 1)).reshape(-1)
            tw = numtools.model_inverse_utility(v_interp, cfg.b, cfg.theta, use_crra)
            v_pos = v_interp.copy()
            over_cap = next_state[nbri, 3] > arrays['smaxv'][3]
            if np.any(over_cap):
                v_pos[over_cap] = numtools.model_utility(
                    tw[over_cap] + (growth_m[nbri][over_cap] ** (cfg.T - t)) * gd[over_cap],
                    cfg.b,
                    cfg.theta,
                    use_crra,
                )
            v_flat[nbri] = v_pos
        neg = np.where(~pos)[0]
        if neg.size > 0:
            v_flat[neg] = numtools.model_utility(
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

        max_rows = int(getattr(cfg, 'two_stage_fine_max_rows', getattr(cfg, 'q_batch_max_rows', 100_000)))
        fine_batch = int(getattr(cfg, 'two_stage_fine_state_batch', 0))
        if fine_batch <= 0:
            fine_batch = max(1, max_rows // max(1, fine_nq))

        for batch_start in range(0, nn, fine_batch):
            batch_end = min(batch_start + fine_batch, nn)
            batch_size = batch_end - batch_start

            state_batch = state[batch_start:batch_end]
            state_batch_rep = np.repeat(state_batch[None, :, :], fine_nq, axis=0).reshape(-1, state.shape[1])

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
    arrays = utils.build_model_arrays(cfg)
    n = arrays['n']
    shock_nodes, shock_weights = numtools.gauss_hermite_multinormal(arrays['m'], arrays['Ee'], arrays['VarCov'])
    state_nodes = [numtools.uniform_nodes(int(n[d]), float(arrays['smin'][d]), float(arrays['smax'][d])) for d in range(len(n))]
    inverse_basis = [
        np.linalg.inv(numtools.linear_spline_basis(int(n[d]), float(arrays['smin'][d]), float(arrays['smax'][d]), state_nodes[d]))
        for d in range(len(n))
    ]
    state_grid = numtools.cartesian_grid(state_nodes)

    n_points = state_grid.shape[0]
    coeffs_over_time = np.zeros((n_points, cfg.T), dtype=float)
    active_idx = np.where(state_grid[:, 3] > 1.0)[0]
    active_states = state_grid[active_idx]
    coeffs = None
    use_crra = bool(getattr(cfg, "CRRA", True))

    for t in range(cfg.T, cfg.t1 - 1, -1):
        _, _, value = mutual_vmaxh(active_states, coeffs, t, cfg, arrays, shock_nodes, shock_weights)
        value_full = np.full(n_points, numtools.model_utility(0.0, cfg.b, cfg.theta, use_crra), dtype=float)
        value_full[active_idx] = value
        coeffs = numtools.apply_inverse_basis_chain(inverse_basis, value_full.reshape(-1, 1)).reshape(-1)
        coeffs_over_time[:, t - 1] = coeffs

    return coeffs_over_time