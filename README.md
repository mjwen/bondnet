Use the pretrained model: [![Binder](https://mybinder.org/badge_logo.svg)](https
://mybinder.org/v2/gh/mjwen/bondnet/binder?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)
Train the model: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/binder?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)



# Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Use pretrained model for prediction](#use-pretrained-model-for-prediction)
- [Train the model](#train-the-model)


BonDNet is a graph neural network model for the prediction of bond dissociation energies (BDEs). It can be applied to both homolytic and heterolytic bond dissociations for molecules of any charge.




<a name="installation"></a>
# Installation

Currently, we only support installation from source:

1. create a conda environment
    ```bash
    conda create --name bondnet
    conda activate bondnet
    conda install python==3.7.4
    ```

2. install dependencies (see their websites for more options)
    ```bash
    conda install pytorch==1.5.0 torchvision -c pytorch   # https://pytorch.org
    conda install dgl==0.4.3 -c dglteam                   # https://www.dgl.ai/pages/start.html
    conda install pymatgen==2020.6.8 -c conda-forge       # https://pymatgen.org/installation.html
    conda install rdkit==2020.03.3 -c conda-forge         # https://rdkit.org/docs/Install.html
    conda install openbabel==3.1.1 -c conda-forge         # http://openbabel.org/wiki/Category:Installation
    ```

3. install this repo
    ```bash
    git clone https://github.com/mjwen/bondnet.git
    pip install -e bondnet
    ```

<a name="use-pretrained-model-for-prediction"></a>
# Use pretrained model for prediction

For a quick prediction of the BDEs for a single molecule, try the live demo at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/binder?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)

A command line interface (CLI) `bondnet` is also provided to make batch predictions.
We support multiple modes of predictions using different input data format.

- a single molecule given by a `SMILES` or `InChI` string, e.g.:
    ```bash
    $ bondnet single "C1COC(=O)O1"
    ```

- multiple molecules listed in a file. Supported molecule format incldues `sdf`, `pdb`, `smiles` and `inchi`, e.g.:
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

Descriptions of the format of the input files and examples can be found
[here](https://github.com/mjwen/bondnet/tree/binder/bondnet/scripts/examples/predict).


<a name="train-the-model"></a>
# Train the model

The [train_bde.ipynb]() Jupyter notebook shows how to train BonDNet on an BDE dataset of both neutral and charged molecules. Try it at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/master?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)

The [train_bde.ipynb]() Jupyter notebook trains a model on CPU. If you want to train on GPUs (a single GPU or distributed), take a look at [train_bde_distributed.py]().

Descriptions of the format of the input files for training and examples can be found
[here](https://github.com/mjwen/bondnet/tree/master/bondnet/scripts/examples/train).

