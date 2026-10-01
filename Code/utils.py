import numpy as np
import pathlib as Path
import csv
import math
from dataclasses import dataclass


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
    ## Parameters for the mutual fund model
    beta0: float = 1.51181
    beta1: float = 0.742391
    alpha0: float = 0.215
    alpha1: float = 0.908361
    alpha2: float = 0.079432
    ## Parameters for the shock distribution
    Ee: tuple[float, float] = (0.0, 0.0)
    var_cov: tuple[tuple[float, float], tuple[float, float]] = ((0.030186, 0.0), (0.0, 0.017292))
    cost: float = 231.0 # operating cost of farmland
    cost_scale_eta: float = 1.0 # eta parameter for the operating cost
    cost_scale_ref: float = 600.0 # reference scale for the operating cost
    sminv: tuple[float, float, float, float] = (230.0, 1010.0, 400.0, 0.0) # minimum values for the state variables
    smaxv: tuple[float, float, float, float] = (540.0, 2840.0, 2000.0, 6_000_000.0) # maximum values for the state variables
    rp: float = 0.03 # risk-premium interest rate
    rn: float = 0.06 # risk-neutral interest rate
    tcs: float = 0.06 # transaction cost of selling farmland
    tcb: float = 0.01 # transaction cost of buying farmland
    fme: float = 300.0 # fixed maintenance cost of farmland
    fds: float = 0.07 # fixed depreciation cost of farmland
    dau: float = 0.7 # rho
    wealth_min: float = 100.0 
    wealth_max: float = 5_000.0
    wealth_size: int = 250
    action_grid_step: float = 0.1 
    risk_free_gross: float | None = None # risk-free gross return
    data_dir: str = "Data"

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

def choose_value_maximizer(cfg):
    return "vmaxh1" if cfg.qf == 0 else "vmaxh2"


def _value_backend(cfg):
    return str(getattr(cfg, 'value_backend', 'numba')).lower()