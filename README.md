[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/gnn/binder)

# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

<a name="installation"></a>
# Installation

Currently, we only support installation from source:

1. create a conda environment
    ```bash
    conda create --name bdenet
    conda activate bdenet
    conda install python==3.7.4
    ```

2. install dependencies (see their websites for more installing options)
    ```bash
    conda install pytorch torchvision -c pytorch       # https://pytorch.org
    conda install dgl -c dglteam                       # https://www.dgl.ai/pages/start.html
    conda install pymatgen==2019.10.4 -c conda-forge   # https://pymatgen.org/installation.html
    conda install rdkit==2020.03.1 -c conda-forge      # https://rdkit.org/docs /Install.html
    conda install openbabel==2.4.1 -c conda-forge      # http://openbabel.org/wiki/Category:Installation
    ```

   NOTE (Latest pymatgen depends on openbabel 3.0.0, which introduces
    many backward incompatible API changes (bad decision). Remember to update the code
     and switch to use the latest.)

3. install this repo
    ```bash
    git clone -b stable https://github.com/mjwen/gnn.git
    pip install -e gnn
    ```

<a name="usage"></a>
# Usage

## Use pretrained model for prediction

We support multiple formats in predicting the bond dissociation energies.
The prediction can be made by running the script
[predict_gated_electrolyte_rxn_ntwk.py](https://github.com/mjwen/gnn/blob/stable/gnn/predict_gated_electrolyte_rxn_ntwk.py)
at: `gnn/gnn/predict_gated_electrolyte_rxn_ntwk.py`.

1. (to come) all bonds in a molecule given by a smiles string, e.g.:
    ```bash
    python predict_gated_electrolyte_rxn_ntwk.py -m C1COC(=O)O1
    ```
2. smiles molecules given in a csv file
(e.g. [smiles_reactions.csv](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples/smiles_reactions.csv).
 See [README](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples/README.md)
   there for its format.)
    ```bash
    python predict_gated_electrolyte_rxn_ntwk.py  -i smiles_reactions.csv  -o results.csv
    ```

3. molecules in an sdf file, charges in a plain text file, and reactions in a csv file
 (see [here](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples) for
  examples and the [README](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples/README.md)
   there for formats of the files):
    ```bash
    python predict_gated_electrolyte_rxn_ntwk.py  -t sdf  -i molecules.sdf charges.txt reactions.csv  -o results.csv
    ```
4. molecule graphs in a json (or yaml) file, and reactions in a csv file
 (see [here](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples) for
  examples and the [README](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples/README.md)
   there for formats of the files):
    ```bash
    python predict_gated_electrolyte_rxn_ntwk.py  -t graph  -i molecule_graphs.json reactions.csv  -o results.csv
    ```

## (to come) Train the model for your own dataset
