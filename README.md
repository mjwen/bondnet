[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/binder?filepath=bondnet%2Fscripts%2Fprediction_binder.ipynb)

# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

<a name="installation"></a>
# Installation

Currently, we only support installation from source:

1. create a conda environment
    ```bash
    conda create --name bondnet
    conda activate bondnet
    conda install python==3.7.4
    ```

2. install dependencies (see their websites for more installing options)
    ```bash
    conda install pytorch==1.5.0 torchvision -c pytorch   # https://pytorch.org
    conda install dgl==0.4.3 -c dglteam                   # https://www.dgl.ai/pages/start.html
    conda install pymatgen==2020.6.8 -c conda-forge       # https://pymatgen.org/installation.html
    conda install rdkit==2020.03.3 -c conda-forge         # https://rdkit.org/docs/Install.html
    conda install openbabel==3.1.1 -c conda-forge         # http://openbabel.org/wiki/Category:Installation
    ```

3. install this repo
    ```bash
    git clone -b stable https://github.com/mjwen/bondnet.git
    pip install -e bondnet
    ```

<a name="usage"></a>
# Usage

## Use pretrained model for prediction

For a quick prediction of the energies of bonds in a single molecule, try the live demo at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/binder?filepath=bondnet%2Fscripts%2Fprediction_binder.ipynb)

A command line interface (CLI) `bondnet` is also provided to make batch predictions. 
We support multiple modes of predictions using different input data format.

- a single molecule given by a `SMILES` or `InChI` string, e.g.:
    ```bash
    $ bondnet single "C1COC(=O)O1"
    ```

- multiple molecules listed in a file. Supported molecule format incldues `sdf`, `pdb
`, `smiles` and `inchi`, e.g.:
    ```bash
    $ bondnet multiple molecules.sdf -o results.sdf
    ```
  
- explicitly specifying the bonds to predict. In this mode, a `moleclue` file
  to list all the molecules and a `reaction` file to list all the reactions associated
  with the bonds are needed (Optionally, a `charge` file lists the charges of all the
  molecules can be provided). Supported molecule format includes `graph `, `sdf`, `pdb`, 
  `smiles `, and `inchi`. e.g.
    ```bash
    $ bondnet reaction -t sdf molecules.sdf reactions.csv -c charges.txt
    ```  
    ```bash
    $ bondnet reaction -t graph molecule_graphs.json reactions.csv
    ```  

Detailed description of the format of the input files and examples can be found 
[here](https://github.com/mjwen/bondnet/tree/binder/bondnet/prediction/examples).


## (to come) Train the model for your own dataset
