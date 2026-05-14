## Dynamic Programming Term Paper

This repository contains the code and term paper for the course **Dynamic Programming and Structural Econometrics** at the University of Copenhagen.

## Python library layout

The core implementation is split into four modules (import names match the filenames):

| Module | File | Role |
|--------|------|------|
| **utils** | `utils.py` | Configuration (`PortfolioConfig`), CSV data loading, balance-sheet helpers (e.g. DFA), and shared array builders (`build_model_arrays`, `choose_value_maximizer`, `_value_backend`). |
| **numerical tools** | `numerical_tools.py` | Quadrature and grids (`gauss_hermite_multinormal`, `uniform_nodes`, `cartesian_grid`), spline / tensor interpolation, utility transforms, and Numba-accelerated scalar helpers used by the solvers. |
| **model without mutual fund** | `model1.py` | Farmland trade only: state transitions, Bellman evaluation, value maximization, `solve_model_coefficients`, `compute_policy_grid_local`, and `PortfolioChoiceModel` (reduced wealth-grid problem). Depends on **utils** and **numerical tools**. |
| **model with mutual fund** | `model2.py` | Adds the mutual fund: `MutualFundConfig`, `MutualFundModel`, `mutual_transition_next_state`, `mutual_vmaxh`, `solve_mutual_coefficients`. Depends on **utils** and **numerical tools**. |

The monolithic reference implementation previously lived in `portfolio_core.py`; an archived copy is under `Old gold/` for comparison only.

## Repository contents

```text
model_without_mutual.ipynb   # Replication: model without mutual fund
model_with_mutual.ipynb      # Replication: model with mutual fund
estimate_state_equations.ipynb  # Estimation / auxiliary analysis
utils.py                     # Utils (config, data, shared helpers)
numerical_tools.py           # Numerical tools (quadrature, splines, Numba helpers)
model1.py                    # Model without mutual fund
model2.py                    # Model with mutual fund
```

## Requirements

The code is written in Python 3.12.7. We recommend using Python 3.12.7 or newer, as older versions may not support all syntax used in the project.

Install dependencies:

```bash
pip install numpy matplotlib pandas numba
```

Run notebooks from the repository root (or add this folder to `PYTHONPATH`) so `utils`, `numerical_tools`, `model1`, and `model2` import correctly.

## Running the code

- **`model_without_mutual.ipynb`** — solves and plots the variant **without** a mutual fund (imports from `utils`, `numerical_tools`, and `model1`).
- **`model_with_mutual.ipynb`** — solves and plots the variant **with** a mutual fund (imports from `utils`, `numerical_tools`, and `model2`).
- **`estimate_state_equations.ipynb`** — supporting estimation workflow as needed for the paper.

Each notebook’s first code cell lists the concrete imports; you do not need `model2` in the no-mutual notebook, or `model1` in the mutual notebook, unless you intentionally combine both.

## Authors

Mikkel Foss Engelsted (hrx712)

Mikkel Rath Tornerup (xqt272)

Nicklas Busk Jensen (vhr863)

## Supervisor

Bertel Schjerning and Max Blesch
