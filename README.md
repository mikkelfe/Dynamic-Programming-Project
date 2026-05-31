## Dynamic Programming Term Paper
This repository contains the code and term paper for the course **Dynamic Programming and Structural Econometrics** at the University of Copenhagen.

The project studies a dynamic portfolio choice model inspired by **Lohano and King (2009)**. The model is used to analyze how a farmer chooses between farmland, financial assets, and debt under uncertainty.

## Repository contents
The main files and folders are:

- **Paper.pdf** (Our paper)
- **model1.ipynb** (Runs the model and simulations for model without mutual fund, and produces all tables and figures.)
- **model2.ipynb** (Runs the model and simulations for model with mutual fund, and produces all tables and figures.)
- **numerical_tools.py** (Library with the core numerical tools used to solve model)
- **utils.py** (Library with util functions used)
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
To run the code, please run the the main notebook **Finished_model.ipynb**

The code produces all model solutions, tables, and figures used in the paper.

## Authors
Mikkel Foss Engelsted (hrx712) 

Mikkel Rath Tornerup (xqt272) 

Nicklas Busk Jensen (vhr863)

## Supervisors
Bertel Schjerning & Max Blesch
