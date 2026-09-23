


## Dynamic Programming Term Paper
This repository contains the code and term paper for the course **Dynamic Programming - Theory, Computation, and Empirical Applications** at the University of Copenhagen.

The project studies a dynamic portfolio choice model inspired by **Lohano and King (2009)**. The model is used to analyze how a farmer chooses between farmland, financial assets, and debt under uncertainty.

The grade we recieved was 12/12 on the danish 7 point grading scale.

## Abstract
This paper replicates and extends the stochastic dynamic programming model of
farmland investment and portfolio choice by Lohano and King (2009). The model
studies how farmers optimally allocate wealth between farmland, financial assets,
borrowing, and risk-free savings under uncertainty regarding farming returns, farm-
land prices, and financial market returns. We implement the original model in
Python and reproduce the main policy functions and simulation results from the
original study, including optimal farmland investment decisions, wealth accumula-
tion, and probabilities of exiting farming.

Using the replicated framework, we conduct counterfactual analyses to examine the
roles of borrowing constraints, transaction costs, and economies of scale. The results
show that relaxing borrowing constraints mainly increases the scale and riskiness of
investment decisions, while transaction costs generate inaction in farmland adjust-
ment. When transaction costs are removed, land trading becomes more active and
long-run farmland holdings increase. Furthermore, we extend the model by intro-
ducing economies of scale in production costs, which significantly increases optimal
farm size, expected wealth, and farm survival rates, while also increasing wealth
dispersion across farmers.

## Repository contents
The main files and folders are:

- **Dynamic_Programming_Paper_2026.pdf** (The final paper)
- **model_with_mutual.ipynb** (Runs the model and simulations for model without mutual fund, and produces all tables and figures.)
- **model_without_mutual.ipynb** (Runs the model and simulations for model without mutual fund, and produces all tables and figures.)
- **model2.py** (Constains the code for setting up model with mutual fund.)
- **model1.py** (Constains the code for setting up model without mutual fund.)
- **numerical_tools.py** (Library with the core numerical tools used to solve model.)
- **utils.py** (Library with util functions used.)
- **estimate_state_equations.ipynb** (Notebook estimating and state equations.)
- **Graphs/** (Folder with our output figures.)
- **Tables/** (Folder with our output tables.)
- **Data/**  (Folder with the data used, provided to us by Lohano and King.)  

## Requirements
The code is written in Python 3.12.7. We recommend using Python 3.12.7 or newer, as older versions may not support all syntax used in the project.

The following packages are required to run the code:
```bash
pip install numpy matplotlib pandas numba
```

## Running the code
To run the code, please run the the two main notebooks **model_with_mutual.ipynb** and **model_without_mutual.ipynb**.

The code produces all model solutions, tables, and figures used in the paper.

## Authors
Mikkel Foss Engelsted (hrx712) 

Mikkel Rath Tornerup (xqt272) 

Nicklas Busk Jensen (vhr863)

## Supervisors
Bertel Schjerning & Max Blesch
