import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from bondnet.data.qm9 import QM9Dataset, create_edge_label_based_on_bond
from bondnet.utils import expand_path, create_directory
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    MolWeightFeaturizer,
)


def load_mols(filename):
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


def plot_bond_distance_hist(
    # filename="~/Documents/Dataset/qm9/gdb9_n200.sdf",
    filename="~/Documents/Dataset/qm9/gdb9.sdf",
):
    """
    Plot the bond distance hist.
    """

    def plot_hist(data, filename):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(data, 20)

        ax.set_xlabel("Bond length")
        ax.set_ylabel("counts")

        fig.savefig(filename, bbox_inches="tight")

    def get_distances(m):
        num_bonds = m.GetNumBonds()
        dist = []
        for u in range(num_bonds):
            bond = m.GetBondWithIdx(u)
            at1 = bond.GetBeginAtomIdx()
            at2 = bond.GetEndAtomIdx()
            atoms_pos = m.GetConformer().GetPositions()
            length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
            dist.append(length)
        return dist

    # prepare data
    mols = load_mols(filename)
    data = [get_distances(m) for m in mols]
    data = np.concatenate(data)

    print("\n\n### atom distance min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/db_access/qm9/bond_distances.pdf"
    create_directory(filename)
    filename = expand_path(filename)
    plot_hist(data, filename)


def get_dataset_qm9(
    # sdf_file="~/Documents/Dataset/qm9/gdb9_n200.sdf",
    # label_file="~/Documents/Dataset/qm9/gdb9_n200.sdf.csv",
    sdf_file="~/Documents/Dataset/qm9/gdb9.sdf",
    label_file="~/Documents/Dataset/qm9/gdb9.sdf.csv",
):
    """
    By running this, we observe the output to get a sense of the low and high values
    for bond length featurizer.

    """
    grapher = HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondAsNodeFeaturizer(
            # length_featurizer="bin",
            # length_featurizer_args={"low": 0.7, "high": 2.5, "num_bins": 10},
            length_featurizer="rbf",
            length_featurizer_args={"low": 0.3, "high": 2.3, "num_centers": 20},
        ),
        global_featurizer=MolWeightFeaturizer(),
        self_loop=True,
    )

    dataset = QM9Dataset(
        grapher=grapher, sdf_file=sdf_file, label_file=label_file, properties=["u0_atom"]
    )

    return dataset


def write_edge_label_based_on_molecule_bond():
    sdf_file = "~/Documents/Dataset/qm9/gdb9_n200.sdf"

    create_edge_label_based_on_bond(
        sdf_file,
        sdf_filename="/Users/mjwen/Applications/db_access/qm9/struct_gdb9_bond_annotation.sdf",
        label_filename="/Users/mjwen/Applications/db_access/qm9/label_gdb9_bond_annotation.yaml",
        feature_filename="/Users/mjwen/Applications/db_access/qm9/feature_gdb9_bond_annotation.yaml",
    )


if __name__ == "__main__":
    plot_bond_distance_hist()
    # get_dataset_qm9()
    # write_edge_label_based_on_molecule_bond()
