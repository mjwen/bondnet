# Use pretrained model

- [Pretrained models](#pretrained-models)
- [Make predictions for all bonds in molecules](#make-predictions-for-all-bonds-in-molecules)
  - [A single molecule](#a-single-molecule)
  - [Multiple molecules](#multiple-molecules)
- [Make predictions for bonds specified in reactions](#make-predictions-for-bonds-specified-in-reactions)
  - [Molecules given as pymatgen MoleculeGraph](#molecules-given-as-pymatgen-moleculegraph)
  - [Molecules given as SDF, PDB, or SMILES](#molecules-given-as-sdf-pdb-or-smiles)

## Pretrained models

Two pretrained models are available for use: one trained on the `mesd` BDE da taset of
homolytic and heterolytic bond dissociations for molecules of charge -1, 0, and  1;
the other trained on the `pubchem` BDE dataset of homolytic bond dissociation of
neutral molecules. The default model is `pubchem` and models can be switched by providing
the dataset name (`mesd` or `pubchem`).

## Make predictions for all bonds in molecules

For a given molecule, all its bonds are broken to create a number reactions.
For a given bond, all possible charge combinations for the products are considered
and the smallest energy is reported. Take the `C-O` bond in `H3C-OH` as an example.
Assume we use the `mesd` model and `H3C-OH` has a charge of `0`, then there are three
reactions because molecules are allowed to have charge -1, 0, and 1 in the `mesd` model:

- `H3C-OH (0) --> H3C (0)  + OH (0)`
- `H3C-OH (0) --> H3C (-1)  + OH (1)`
- `H3C-OH (0) --> H3C (1)  + OH (-1)`

The energy of all three reactions are computed and the smallest is reported.

Since the `pubchem` model only allows neutral molecules, there is always only one
reaction for each bond.

### A single molecule

- Try the live demo at: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mjwen/bondnet/pretrained?filepath=bondnet%2Fscripts%2Fpredict_binder.ipynb)

- Or use the `bondnet` command line tool and provide the molecule as a `SMILES` or
    `InChI` string.

    Example:
    ```bash
    bondnet single "C1COC(=O)O1"    # default to use pubchem model
    bondnet single --ring-bond  "C1COC(=O)O1"
    bondnet --model bdncm single "C1COC(=O)O1"
    bondnet --model bdncm single --charge -1 "C1COC(=O)O1"
    bondnet single -h
    ```

### Multiple molecules

The molecules should be given in a  file (e.g. [molecules.sdf](./molecules.sdf) and
[molecules.smi](./molecules.smi)).
Supported molecule format includes `sdf`, `pdb`, `smiles` and `inchi`.

Example:
```bash
bondnet multiple molecules.sdf -o results.sdf
bondnet multiple -t smiles molecules.smi -o results.sdf
bondnet multiple -h
```

## Make predictions for bonds specified in reactions

If you know the bond dissociation reactions (including charges of reactants and products),
the bond dissociation energies can be directly predict for them. Depending on how you want
to specify the molecules, different inputs are needed.


### Molecules given as pymatgen MoleculeGraph

Two files are needed:

1. pymatgen molecule graph file (e.g.[molecule_graphs.yaml](./molecule_graphs.yaml)
    and [molecule_graphs.json](./molecule_graphs.json))

    A yaml or json file listing all the molecule graphs. Each graph is given as a
    dictionary that can be obtained by `MoleculeGraph.as_dict()`. The molecules are
    indexed as `0, 1, 2,...` (from top to bottom).

5. reaction file (e.g. [reactions.csv](./reactions.csv))

    A csv file listing the reactions. Each line should list a reaction in the format:

    ```reactant,product1,product2```

    where `reactant`, `product1`, and `product2` are indices of molecules in the
    molecule file. If there is only one product, `product2` is left blank and this
    becomes (don't forget the trailing comma):

    ```reactant,product1,```

Exmaple:
```bash
bondnet reaction -t graph molecule_graphs.json reactions.csv
bondnet --model bdncm reaction -t graph molecule_graphs.json reactions.csv
bondnet reaction -h
```

### Molecules given as SDF, PDB, or SMILES

Three files are needed:

1. molecule file (e.g. [molecules.sdf](./molecules.sdf) and [molecules.smi](./molecules.smi))

    An SDF, PBD, or SMILES file listing all the molecules.
    The molecules are indexed as: `0, 1, 2, ...` (from top to bottom).

2. reaction file (e.g. [reactions.csv](./reactions.csv))

    Same as the format specified in [Molecules given as pymatgen MoleculeGraph](#molecules-given-as-pymatgen-moleculegraph)

3. (optional) charge file (e.g. e.g. [charges.txt](./charges.txt))

    A plain text file listing the charge of molecules given in the molecule file.
    If this file is is not provided, all molecules are treated as neutral.

Example:
```bash
bondnet reaction -t sdf molecules.sdf reactions.csv
bondnet --model bdncm reaction -t smiles molecules.smi reactions.csv -c charges.txt
```
