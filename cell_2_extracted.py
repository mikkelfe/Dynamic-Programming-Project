from dataclasses import dataclass

import numpy as np

from dynprolib.pyfriendly import PortfolioConfig
from dynprolib.pyfriendly.portfolio_core import (
    apply_inverse_basis_chain,
    build_model_arrays,
    cartesian_grid,
    gauss_hermite_multinormal,
    linear_spline_basis,
    model_inverse_utility,
    model_utility,
    tensor_basis_interpolate,
    uniform_nodes,
)


@dataclass
class MutualFundConfig(PortfolioConfig):
    theta: float = 1.0
    gamma0: float = 0.057757
    EgRM: float = 1.073826
    n: tuple[int, int, int, int] = (7, 7, 5, 21)
    q: int = 41
    q2: int = 25
    qf: int = 21
    search_mode: str = "matlab"
    value_backend: str = "numba"
    q_batch: int = 4
    q_batch_max_rows: int = 100_000
    q_coarse_step: int = 4
    q_refine_radius: int = 2
    m: tuple[int, int, int] = (3, 3, 3)
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
    if np.all(state[:, 2] + trade_x < 1.0):
        xml = state[:, 3].copy()
        xmu = state[:, 3].copy()
        return xml, xmu

    s2bs = (1.0 + cfg.tcb) * state[:, 1] + cfg.fme
    sell_mask = trade_x < 0
    s2bs[sell_mask] = (1.0 - cfg.tcs) * state[sell_mask, 1] + (1.0 - cfg.fds) * cfg.fme
    s2v = (1.0 - cfg.tcs) * state[:, 1] + (1.0 - cfg.fds) * cfg.fme
    at = state[:, 3] - s2v * state[:, 2]
    xml = np.zeros(state.shape[0], dtype=float)
    xmu = np.maximum(0.0, at - s2bs * trade_x + cfg.dau * s2v * (state[:, 2] + trade_x))
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

    if np.all(low_land):
        next_state[:, 3] = cfg.EgRM * state[:, 3]
        return next_state

    cash_after_cost = equity_now - buy_price * trade_x - cfg.cost * next_state[:, 2] - mutual_x
    rate_high_land = np.full(state.shape[0], cfg.rp, dtype=float)
    rate_high_land[cash_after_cost < 0] = cfg.rn
    wealth_high_land = (1.0 + rate_high_land) * cash_after_cost + next_state[:, 0] * next_state[:, 2] + sell_price_next * next_state[:, 2] + growth_m * mutual_x

    next_state[:, 3] = wealth_high_land
    return next_state


def _search_mode(cfg):
    return str(getattr(cfg, 'search_mode', 'matlab')).lower()


def _value_backend(cfg):
    return str(getattr(cfg, 'value_backend', 'numba')).lower()


from numba import njit


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
            value[i] = _numba_model_utility_scalar(cont_factor * W, b, theta, crra)
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


def mutual_value_function(state, coeffs, t, controls, cfg, arrays, shock_nodes, shock_weights):
    state = np.asarray(state, dtype=float)
    controls = np.asarray(controls, dtype=float)
    trade_x = controls[:, 0]
    mutual_x = controls[:, 1]
    use_crra = bool(getattr(cfg, "CRRA", True))
    mode = _search_mode(cfg)
    backend = _value_backend(cfg)

    if backend == 'numba' and coeffs is None and mode != 'matlab':
        return _numba_mutual_value_function_no_coeffs(
            np.ascontiguousarray(state, dtype=np.float64),
            np.ascontiguousarray(trade_x, dtype=np.float64),
            np.ascontiguousarray(mutual_x, dtype=np.float64),
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
            np.ascontiguousarray(shock_nodes, dtype=np.float64),
            np.ascontiguousarray(shock_weights, dtype=np.float64),
        )

    if mode == 'matlab':
        value = np.zeros(state.shape[0], dtype=float)
        for k in range(shock_nodes.shape[0]):
            shocks = np.tile(shock_nodes[k], (state.shape[0], 1))
            next_state = mutual_transition_next_state(state, trade_x, mutual_x, shocks, cfg, arrays['sminv'], arrays['smaxv'])
            v_k = np.zeros(state.shape[0], dtype=float)
            low_land = next_state[:, 2] < 1.0

            if np.all(low_land):
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
                            v_nbri[over_cap] = model_utility(
                                tw[over_cap] + ((1.0 + cfg.rp) ** (cfg.T - t)) * gd[over_cap],
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
                    tw[over_cap] + ((1.0 + cfg.rp) ** (cfg.T - t)) * gd[over_cap],
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

    xmq = xml[None, :] + gap[None, :] * np.arange(cfg.q2, dtype=float)[:, None]
    state_rep = np.repeat(state[None, :, :], cfg.q2, axis=0).reshape(-1, state.shape[1])
    trade_rep = np.repeat(xqi[None, :], cfg.q2, axis=0).reshape(-1)
    mutual_rep = xmq.reshape(-1)
    controls_rep = np.column_stack([trade_rep, mutual_rep])
    value_rep = mutual_value_function(state_rep, coeffs, t, controls_rep, cfg, arrays, shock_nodes, shock_weights)
    vxq2 = value_rep.reshape(cfg.q2, nn)
    ind = np.argmax(vxq2, axis=0)
    j = np.arange(nn)
    xmqi = xmq[ind, j]
    vqi = vxq2[ind, j]
    return xmqi, vqi


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
            q_batch = 1

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
        vx_coarse, _ = _evaluate_q_subset(coarse_idx)
        coarse_best_pos = np.argmax(vx_coarse, axis=0)
        coarse_best_idx = coarse_idx[coarse_best_pos]
        j_idx = np.arange(nn)
        x_coarse = xq[coarse_best_idx, j_idx]
        xas = 1600.0 / max(1.0, (cfg.q - 1))
        xlf = np.maximum(xl, x_coarse - xas)
        xuf = np.minimum(xu, x_coarse + xas)
        fine_q = int(getattr(cfg, 'qf', cfg.q))
        fine_nq = fine_q + 2
        
        best_v = np.full(nn, -np.inf, dtype=float)
        best_x = np.zeros(nn, dtype=float)
        best_xm = np.zeros(nn, dtype=float)
        
        # Batch process fine search to avoid allocating entire state array at once
        fine_batch = max(1, int(getattr(cfg, 'q_batch', 4)))
        max_rows = int(getattr(cfg, 'q_batch_max_rows', 100_000))
        if nn * fine_nq * fine_batch > max_rows:
            fine_batch = max(1, max_rows // (nn * fine_nq))
        
        for batch_start in range(0, nn, fine_batch):
            batch_end = min(batch_start + fine_batch, nn)
            batch_size = batch_end - batch_start
            
            state_batch = state[batch_start:batch_end]
            state_batch_rep = np.repeat(state_batch, fine_nq, axis=0)
            x_batch_ref = np.zeros(batch_size * fine_nq, dtype=float)
            
            ptr = 0
            for j_local in range(batch_size):
                j_global = batch_start + j_local
                x_candidates = np.empty(fine_nq, dtype=float)
                x_candidates[0] = 0.0
                x_candidates[1] = -state[j_global, 2]
                if fine_q <= 1:
                    gap_fine = 0.0
                else:
                    gap_fine = (xuf[j_global] - xlf[j_global]) / (fine_q - 1)
                idxs = np.arange(fine_nq - 2, dtype=float)
                x_candidates[2:] = xlf[j_global] + idxs * gap_fine
                x_batch_ref[ptr:ptr + fine_nq] = x_candidates
                ptr += fine_nq
            
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
            
            # Track best results for this batch
            for i in range(batch_size * fine_nq):
                j_local = i // fine_nq
                j_global = batch_start + j_local
                if v_batch[i] > best_v[j_global]:
                    best_v[j_global] = v_batch[i]
                    best_x[j_global] = x_batch_ref[i]
                    best_xm[j_global] = xm_batch[i]
        
        return best_x, best_xm, best_v

    full_idx = np.arange(nq, dtype=int)
    vxq, xmq = _evaluate_q_subset(full_idx)
    ind = np.argmax(vxq, axis=0)
    j = np.arange(nn)
    return xq[full_idx[ind], j], xmq[ind, j], vxq[ind, j]