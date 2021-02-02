Use the pretrained model: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)

Train the model: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)


# Table of Contents
- [Installation](#installation)
- [Use pretrained model for prediction](#use-pretrained-model-for-prediction)
- [Train the model](#train-the-model)


BonDNet is a graph neural network model for the prediction of bond dissociation
energies (BDEs). It can be applied to both homolytic and heterolytic bond dissociations
for molecules of any charge. This model is described in the paper: 
[BonDNet: a graph neural network for the prediction of bond dissociation 
  energies for charged molecules, *Chemical Science*, 2021.](https://doi.org/10.1039/D0SC05251E)

<p align="center">
<img src="bondnet.png" alt="BonDNet" width="600">
</p>


# Installation

Currently, we support installation from source:

1. create a conda environment
    ```bash
    conda create --name bondnet
    conda activate bondnet
    conda install python==3.7
    ```

2. install dependencies (see their websites for more options)
    ```bash
    conda install pytorch==1.6.0 torchvision -c pytorch   # https://pytorch.org
    conda install dgl==0.5.0 -c dglteam                   # https://www.dgl.ai/pages/start.html
    conda install pymatgen==2020.8.13 -c conda-forge      # https://pymatgen.org/installation.html
    conda install rdkit==2020.03.5 -c conda-forge         # https://rdkit.org/docs/Install.html
    conda install openbabel==3.1.1 -c conda-forge         # http://openbabel.org/wiki/Category:Installation
    ```

3. install this repo
    ```bash
    git clone https://github.com/mjwen/bondnet.git
    pip install -e bondnet
    ```

# Use pretrained model for prediction

For a quick prediction of the BDEs for a single molecule, try the live demo at:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)

Alternatively, a command line interface (CLI) `bondnet` is provided for batch predictions.
(Optional. We suggest switching to the `pretrained` branch for using the `bondnet` CLI 
to make predictions. It should be more stable. To install the `pretrained` branch
, do ```cd bondnet```, then ```git checkout pretrained```, and finally ```pip install -e
 .```)

- A single molecule given by a `SMILES` or `InChI` string, e.g.:
    ```bash
    bondnet single "C1COC(=O)O1"
    ```

- Multiple molecules listed in a file. Supported molecule format includes `sdf`, `pdb`, `smiles` and `inchi`, e.g.:
    ```bash
    bondnet multiple molecules.sdf -o results.sdf
    ```

- Explicitly specifying the bond dissociation reactions. In this mode, a `moleclue` file
  listing all the molecules and a `reaction` file listing all the bond dissociation 
  reactions are needed. Supported molecule format includes `graph `, `sdf`, `pdb`,
  `smiles `, and `inchi`. e.g.
    ```bash
    bondnet reaction -t graph molecule_graphs.json reactions.csv
    bondnet reaction -t sdf molecules.sdf reactions.csv 
    ```
 
More detailed instructions, example input files, and description of the file formats, 
can be found [here](./bondnet/scripts/examples/predict).


# Train the model

The [train_bde.ipynb](./bondnet/scripts/train_bde.ipynb) Jupyter notebook shows
how to train BonDNet on a BDE dataset of both neutral and charged molecules.
Try it at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)

The [train_bde.ipynb](./bondnet/scripts/train_bde.ipynb) Jupyter notebook trains a model on CPU.
If you want to train on GPUs (a single GPU or distributed), take a look at
[train_bde_distributed.py](./bondnet/scripts/train_bde_distributed.py). A model can be trained by
```bash
python train_bde_distributed.py  molecules.sdf  molecule_attributes.yaml  reactions.yaml
```

More detailed instructions, example input files, and description of the file formats, 
can be found [here](./bondnet/scripts/examples/train).
