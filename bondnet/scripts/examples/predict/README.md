# Format of the files for doing predictions

## Molecules given in SMILES

One csv file is needed (e.g. smiles_reactions.csv). 

- Each line should list a reaction in the format: 

    ```reactant,fragment1,fragment2,charge_reactant,charge_fragment1,charge_fragment2```
    
    where `reactant`, `fragment1`, and `fragment2` are SMILES strings of molecules; and 
    `charge_reactant`, `charge_fragment1`, and `charge_fragment2` are charges of the
     reactant and two fragments, respectively. 
    
- If there is only one fragment (e.g. break a bond in a ring), `fragment2` and 
    `charge_fragment2` should be leave blank, and thus the format becomes 
    (don't forget the trailing comma):
    
    ```reactant,fragment1,,charge_reactant,charge_fragment1,```

- If all your molecules are of charge 0, you can completely skip the charges and then 
    the format becomes:
    
    ```reactant,fragment1,fragment2```
    
    Again, if there is only one fragment, this becomes:
    
    ```reactant,fragment1,```
    
    Note that in the same file you need to either provide charges for all reactions or
    skip them all (if charges are 0), but you cannot mix the two formats.
    
 
 ## Molecules given in SDF

Three files are needed:

1. molecule file (e.g. molecules.sdf)

    An SDF file lists all the molecules, each per block separated by `$$$$`. The
    indices for the molecules are (from top to bottom): `0, 1, 2, ...`

2. charge file (e.g. charges.txt) 

    A plain text file to specify the charges of molecules given in the molecule file. 

3. reaction file (e.g. reactions.csv) 

    A csv file to specify the reactions. Each line should list a reaction in the format: 
    
    ```reactant,fragment1,fragment2```
    
    where `reactant`, `fragment1`, and `fragment2` are indices of molecules in the
    molecule file. 
    If there is only one fragment, this becomes (don't forget the trailing comma):
    
    ```reactant,fragment1,```

    
 ## Molecule Graphs given in json or yaml files
 
Two files are needed:

1. molecule graph file (e.g. molecule_graphs.json, molecule_graphs.yaml) 
   A json or yaml file lists all the molecule graphs. Each graph is represented by
   a dict that can be obtained by `MoleculeGraph.as_dict()`. 

2. reaction file (e.g. reactions.csv) 
    The same as described in `Molecules given in SDF`.
