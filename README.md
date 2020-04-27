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
    conda install python
    ```

2. install dependencies
    ```bash
    conda install pytorch torchvision -c pytorch   # https://pytorch.org
    conda install dgl -c dglteam                   # https://www.dgl.ai/pages/start.html
    conda isntall pymatgen -c conda-forge          # https://pymatgen.org/installation.html
    conda install rdkit -c rdkit                   # https://rdkit.org/docs/Install.html
    conda install openbabel -c openbabel
    ```
   
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

1. smiles molecules and fragments given in a csv file 
(e.g. [smiles_reactions.csv](https://github.com/mjwen/gnn/blob/stable/gnn/pre_trained/examples/smiles_reactions.csv) )
    ```bash
    python predict_gated_electrolyte_rxn_ntwk.py -i smiles_reactions.csv -o results.csv 
    ```

2. (to come) all bonds in a molecule given by a smiles string, e.g.:
    ```bash
    python predict_gated_electrolyte_rxn_ntwk.py -m C1COC(=O)O1  
    ```
3. (to come) molecules in sdf file, and a yaml file specifying the bond breaking reactions


## (to come) Train the model for your own dataset 
