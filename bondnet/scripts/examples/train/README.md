# Train BonDNet

## Input file format

To train BonDNet, three files are needed:

1. molecule file (e.g. [molecules.sdf](./molecules.sdf))

    An SDF file listing all the molecules in the dataset, each per block separated by
     `$$$$`. 
    The molecules are indexed as  `0, 1, 2, ...` (from top to bottom).

2. molecule attributes file (e.g. [molecule_attributes.yaml](./molecule_attributes.yaml))

    A yaml file specifying the charge of molecules given in the molecule file. 

3. reaction file (e.g. [reactions.yaml](./reactions.yaml))

    A yaml file specifying the bond dissociation reactions. For each bond dissociation
     reaction, 
    the `energy`, `reactants`, and `products` and mandatory. The `reactants` and the
     `products` are specified by the indices of the molecules in the molecule file. 
     The `index` to index the reactions is optional. For example, the first reaction given
       in [reactions.yaml](./reactions.yaml) is 
    ```
    - energy: 2.5077022141203997
      index: 0
      products:
      - 186
      - 38
      reactants:
      - 45 
    ```   
    It means the reactant molecule `45` fragments into two product molecules `38` and
     `186`, and the bond dissociation energy for the reaction is 
     `2.5077022141203997`.  

     
## Train the model 

The [train_bde.ipynb](../../train_bde.ipynb) Jupyter notebook shows
how to train BonDNet on a BDE dataset of both neutral and charged molecules.
Try it at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/master?filepath=bondnet%2Fscripts%2Ftrain_bde.ipynb)

The [train_bde.ipynb](../../train_bde.ipynb) Jupyter notebook trains a model on CPU.
If you want to train on GPUs (a single GPU or distributed), take a look at
[train_bde_distributed.py](../../train_bde_distributed.py). A model can be trained by
```bash
$ python  train_bde_distributed.py  molecules.sdf molecule_attributes.yaml reactions.yaml
```


## (optional) Generate atom-mapped reaction

In fact, [train_bde_distributed.py](../../train_bde_distributed.py) will internally
convert the reactions in the reaction file to atom-mapped reactions, so that difference
features can be computed. If you want to do a larger number of trainings for the same
dataset (hyperparameter search for example), you may want to generate the atom-mapped
 reactions once 
and use them for all you runs. This can save you some time.

[create_label_file.py](../../create_label_file.py) can be used to generate the atom-mapped
reactions:
```bash
$ python  create_label_file.py  molecules.sdf molecule_attributes.yaml reactions.yaml reactions_atom_mapped.yaml
```

Then you can train using the atom-mapped reaction file: 
```bash
$ python  train_bde_distributed.py  molecules.sdf molecule_attributes.yaml reactions_atom_mapped.yaml
```
