import numpy as np
from matplotlib import pyplot as plt
from bondnet.utils import expand_path, create_directory
from bondnet.dataset.zinc_bde import read_zinc_bde_dataset
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    MolWeightFeaturizer,
)
from bondnet.data.electrolyte import ElectrolyteBondDataset


def plot_bond_distance_hist(
    filename="~/Documents/Dataset/ZINC_BDE",
    # filename="~/Documents/Dataset/ZINC_BDE_100",
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
        coords = m.coords
        dist = [np.linalg.norm(coords[u] - coords[v]) for (u, v), _ in m.bonds.items()]
        return dist

    # prepare data
    mols, _ = read_zinc_bde_dataset(filename)
    data = map(get_distances, mols)
    data = np.concatenate(list(data))

    print("\n\n### atom distance min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/db_access/zinc_bde/bond_distances.pdf"
    create_directory(filename)
    filename = expand_path(filename)
    plot_hist(data, filename)


def get_dataset_zinc(
    sdf_file="~/Applications/db_access/zinc_bde/zinc_struct_bond_rgrn.sdf",
    label_file="~/Applications/db_access/zinc_bde/zinc_label_bond_rgrn.txt",
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
            length_featurizer_args={"low": 0.3, "high": 2.5, "num_centers": 20},
        ),
        global_featurizer=MolWeightFeaturizer(),
        self_loop=True,
    )

    dataset = ElectrolyteBondDataset(
        grapher=grapher, sdf_file=sdf_file, label_file=label_file
    )

    return dataset


if __name__ == "__main__":
    # plot_bond_distance_hist()

    get_dataset_zinc()
