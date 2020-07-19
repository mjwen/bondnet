"""
QM9 dataset.
"""

import pandas as pd
import numpy as np
import logging
import warnings
import itertools
from collections import OrderedDict
from rdkit import Chem
from bondnet.data.electrolyte import ElectrolyteMoleculeDataset
from bondnet.utils import expand_path, yaml_dump

logger = logging.getLogger(__name__)


class QM9Dataset(ElectrolyteMoleculeDataset):
    """
    The QM9 dataset.
    """

    def _read_label_file(self):
        """
        Returns:
            rst (2D array): shape (N, M), where N is the number of lines (excluding the
                header line), and M is the number of columns (excluding the first index
                column).
            extensive (list): size (M), indicating whether the corresponding data in
                rst is extensive property or not.
        """

        rst = pd.read_csv(self.raw_labels, index_col=0)
        rst = rst.to_numpy()

        h2e = 27.211396132  # Hartree to eV
        k2e = 0.0433634  # kcal/mol to eV

        # supported property
        supp_prop = OrderedDict()
        supp_prop["A"] = {"uc": 1.0, "extensive": True}
        supp_prop["B"] = {"uc": 1.0, "extensive": True}
        supp_prop["C"] = {"uc": 1.0, "extensive": True}
        supp_prop["mu"] = {"uc": 1.0, "extensive": True}
        supp_prop["alpha"] = {"uc": 1.0, "extensive": True}
        supp_prop["homo"] = {"uc": h2e, "extensive": False}
        supp_prop["lumo"] = {"uc": h2e, "extensive": False}
        supp_prop["gap"] = {"uc": h2e, "extensive": False}
        supp_prop["r2"] = {"uc": 1.0, "extensive": True}
        supp_prop["zpve"] = {"uc": h2e, "extensive": True}
        supp_prop["u0"] = {"uc": h2e, "extensive": True}
        supp_prop["u298"] = {"uc": h2e, "extensive": True}
        supp_prop["h298"] = {"uc": h2e, "extensive": True}
        supp_prop["g298"] = {"uc": h2e, "extensive": True}
        supp_prop["cv"] = {"uc": 1.0, "extensive": True}
        supp_prop["u0_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["u298_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["h298_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["g298_atom"] = {"uc": k2e, "extensive": True}

        if self.properties is not None:

            for prop in self.properties:
                if prop not in supp_prop:
                    raise ValueError(
                        "Property '{}' not supported. Supported ones are: {}".format(
                            prop, supp_prop.keys()
                        )
                    )
            supp_list = list(supp_prop.keys())
            indices = [supp_list.index(p) for p in self.properties]
            rst = rst[:, indices]
            convert = [supp_prop[p]["uc"] for p in self.properties]
            extensive = [supp_prop[p]["extensive"] for p in self.properties]
        else:
            convert = [v["uc"] for k, v in supp_prop.items()]
            extensive = [v["extensive"] for k, v in supp_prop.items()]

        if self.unit_conversion:
            rst = np.multiply(rst, convert)

        return rst, extensive


def create_edge_label_based_on_bond(
    filename,
    sdf_filename="mols.sdf",
    label_filename="bond_label.yaml",
    feature_filename="feature.yaml",
):
    """
    For a molecule from SDF file, creating complete graph for atoms and label the edges
    based on whether its an actual bond or not.

    The order of the edges are (0,1), (0,2), ... , (0, N-1), (1,2), (1,3), ...,
    (N-2, N-1), where N is the number of atoms.

    Args:
        filename (str): name of the input sdf file
        sdf_filename (str): name of the output sdf file
        label_filename (str): name of the output label file
        feature_filename (str): name of the output feature file
    """

    def read_sdf(filename):
        filename = expand_path(filename)
        supp = Chem.SDMolSupplier(filename, sanitize=True, removeHs=False)
        all_mols = []
        for i, mol in enumerate(supp):
            if mol is None:
                print("bad mol:", i)
            else:
                all_mols.append(mol)
        print("{} molecules read from sdf file".format(len(all_mols)))
        return all_mols

    def get_bond_label(m):
        """
        Get to know whether an edge in a complete graph is a bond.

        Returns:
            list: bool to indicate whether an edge is a bond. The edges are in the order:
                (0,1), (0,2), ..., (0,N-1), (1,2), (1,3), ..., (N, N-1), where N is the
                number of atoms.
        """

        num_bonds = m.GetNumBonds()
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        num_atoms = m.GetNumAtoms()
        bond_label = []
        for u, v in itertools.combinations(range(num_atoms), 2):
            bond = m.GetBondBetweenAtoms(u, v)
            if bond is not None:
                bond_label.append(True)
            else:
                bond_label.append(False)

        return bond_label

    mols = read_sdf(filename)

    labels = []
    with open(sdf_filename, "w") as f:
        for m in mols:
            labels.append(get_bond_label(m))
            sdf = Chem.MolToMolBlock(m)
            f.write(sdf + "$$$$\n")
    yaml_dump(labels, label_filename)

    # charge is 0 for all mols
    charges = [{"charge": 0} for _ in range(len(labels))]
    yaml_dump(charges, feature_filename)
