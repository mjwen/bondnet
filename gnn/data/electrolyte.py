"""
The electorlyte dataset.

Based on the TencentAlchemy dataset of dgl:
https://docs.dgl.ai/_modules/dgl/data/chem/alchemy.html#TencentAlchemyDataset
"""
import numpy as np
import torch
import os
import pickle
from collections import defaultdict
from dgl.data.chem.utils import mol_to_complete_graph

try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except ImportError:
    pass


class AtomFeaturizer:
    """
    Featurization for all atoms in a molecule. The atom indices will be preserved.
    """

    def __init__(self, species):
        self.species = species

    @property
    def feature_size(self):
        return self._feature_size

    def __call__(self, mol):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
        atom_feats_dict : dict
            Dictionary for atom features
        """
        mol = rdmolops.AddHs(mol, explicitOnly=False)

        atom_feats_dict = defaultdict(list)
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['node_type'].append(atom_type)

            h_u = []
            h_u += [int(symbol == x) for x in self.species]
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in (
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                )
            ]
            h_u.append(num_h)
            atom_feats_dict['n_feat'].append(
                torch.tensor(np.array(h_u).astype(np.float32))
            )

        self._feature_size = len(atom_feats_dict['n_feat'][0])

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'], dim=0)
        atom_feats_dict['node_type'] = torch.tensor(
            np.array(atom_feats_dict['node_type']).astype(np.int64)
        )

        return atom_feats_dict


class BondFeaturizer:
    """
    Featurization for all bonds in a molecule. The bond indices will be preserved.
    """

    def __init__(self):
        pass

    @property
    def feature_size(self):
        return self._feature_size

    def __call__(self, mol, self_loop=False):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        self_loop : bool
            Whether to add self loops. Default to be False.

        Returns
        -------
        bond_feats_dict : dict
            Dictionary for bond features
        """
        mol = rdmolops.AddHs(mol, explicitOnly=False)

        bond_feats_dict = defaultdict(list)

        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        num_atoms = mol.GetNumAtoms()
        assert (
            num_atoms > 1
        ), 'number of atoms < 2; cannot featurize bond edge of molecule'

        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self_loop:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict['e_feat'].append(
                    [
                        float(bond_type == x)
                        for x in (
                            Chem.rdchem.BondType.SINGLE,
                            Chem.rdchem.BondType.DOUBLE,
                            Chem.rdchem.BondType.TRIPLE,
                            Chem.rdchem.BondType.AROMATIC,
                            None,
                        )
                    ]
                )
                bond_feats_dict['distance'].append(np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict['e_feat'] = torch.tensor(
            np.array(bond_feats_dict['e_feat']).astype(np.float32)
        )

        self._feature_size = len(bond_feats_dict['e_feat'][0])

        bond_feats_dict['distance'] = torch.tensor(
            np.array(bond_feats_dict['distance']).astype(np.float32)
        ).reshape(-1, 1)

        return bond_feats_dict


class ElectrolyteDataset:
    """
    The electrolyte dataset for Li-ion battery.

    Args:
        sdf_file (str): path to the sdf file of the molecules. Preprocessed dataset
            can be stored in a pickle file (e.g. with file extension of `pkl`) and
            provided for fast recovery.
        label_file (str): path to the label file. Similar to the sdf_file, pickled file
            can be provided for fast recovery.
    """

    def __init__(self, sdf_file, label_file):
        self.sdf_file = os.path.abspath(sdf_file)
        self.label_file = os.path.abspath(label_file)

        if self._is_pickled(self.sdf_file) != self._is_pickled(self.label_file):
            raise ValueError('sdf file and label file does not have the same format')
        if self._is_pickled(self.sdf_file):
            self._pickled = True
        else:
            self._pickled = False

        self._load()

    @property
    def feature_size(self):
        return self._feature_size

    def _load(self):

        if self._pickled:
            with open(self.sdf_file, 'rb') as f:
                self.graphs = pickle.load(f)
            with open(self.sdf_file, 'rb') as f:
                self.labels = pickle.load(f)
        else:
            print('Start preprocessing dataset ...')

            self.target = pd.read_csv(
                self.label_file, index_col=0, usecols=['mol', 'property_1']
            )
            self.target = self.target[['property_1']]

            self.graphs = []
            self.labels = []

            supp = Chem.SDMolSupplier(self.sdf_file)
            species = self._get_species()
            atom_featurizer = AtomFeaturizer(species)
            bond_featurizer = BondFeaturizer()

            cnt = 0
            dataset_size = len(self.target)
            for mol, label in zip(supp, self.target.iterrows()):
                mol = rdmolops.AddHs(mol, explicitOnly=False)
                cnt += 1
                print('Processing molecule {}/{}'.format(cnt, dataset_size))

                graph = mol_to_complete_graph(
                    mol, atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer
                )
                smile = Chem.MolToSmiles(mol)
                graph.smile = smile
                self.graphs.append(graph)

                label = torch.tensor(np.array(label[1].tolist(), np.float32))
                self.labels.append(label)

            self._feature_size = {
                'atom_featurizer': atom_featurizer.feature_size,
                'bond_featurizer': bond_featurizer.feature_size,
            }

            # save to disk
            with open(os.path.splitext(self.sdf_file)[0] + '.pkl', 'wb') as f:
                pickle.dump(self.graphs, f)
            with open(os.path.splitext(self.label_file)[0] + '.pkl', 'wb') as f:
                pickle.dump(self.labels, f)

        self.set_mean_and_std()
        print(len(self.graphs), "loaded!")

    def _get_species(self):
        suppl = Chem.SDMolSupplier(self.sdf_file)
        system_species = set()
        for mol in suppl:
            atoms = mol.GetAtoms()
            species = [a.GetSymbol() for a in atoms]
            system_species.update(species)
        return list(system_species)

    def set_mean_and_std(self, mean=None, std=None):
        """Set mean and std or compute from labels for future normalization.

        Parameters
        ----------
        mean : int or float
            Default to be None.
        std : int or float
            Default to be None.
        """
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = np.mean(labels, axis=0)
        if std is None:
            std = np.std(labels, axis=0)
        self.mean = mean
        self.std = std

    @staticmethod
    def _is_pickled(fname):
        if os.path.splitext(fname)[1] == 'pkl':
            return True
        else:
            return False

    def __getitem__(self, item):
        """Get datapoint with index

        Args:
            item (int): Datapoint index

        Returns:
            g: DGLGraph for the ith datapoint
            l (Tensor of dtype float32): Labels of the datapoint for all tasks
        """
        g, l = self.graphs[item], self.labels[item]
        return g, l

    def __len__(self):
        """Length of the dataset

        Returns:
            Length of Dataset
        """
        return len(self.graphs)
