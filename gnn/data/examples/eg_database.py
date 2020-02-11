import os
import itertools
import numpy as np
import subprocess
from matplotlib import pyplot as plt
from rdkit import Chem
from gnn.data.database import DatabaseOperation
from gnn.utils import pickle_dump, pickle_load, expand_path


def pickle_db_entries():
    entries = DatabaseOperation.query_db_entries(
        db_collection="mol_builder", num_entries=200
    )
    # entries = DatabaseOperation.query_db_entries(db_collection="smd", num_entries=200)

    filename = "~/Applications/db_access/mol_builder/database_n200.pkl"
    pickle_dump(entries, filename)


def pickle_molecules():
    # db_collection = "task"
    db_collection = "mol_builder"
    entries = DatabaseOperation.query_db_entries(
        db_collection=db_collection, num_entries=None
    )

    mols = DatabaseOperation.to_molecules(entries, db_collection=db_collection)
    filename = "~/Applications/db_access/mol_builder/molecules_unfiltered.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200_unfiltered.pkl"
    pickle_dump(mols, filename)

    mols = DatabaseOperation.filter_molecules(mols, connectivity=True, isomorphism=True)
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    pickle_dump(mols, filename)


def print_mol_property():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    m = mols[10]

    # get all attributes
    for key, val in vars(m).items():
        print("{}: {}".format(key, val))

    # @property attributes
    properties = [
        "charge",
        "spin_multiplicity",
        "atoms",
        "bonds",
        "species",
        "coords",
        "formula",
        "composition_dict",
        "weight",
    ]
    for prop in properties:
        print("{}: {}".format(prop, getattr(m, prop)))

    # using atom and bond
    print("looping m.atoms")
    for idx, attr in m.atoms:
        print(idx, attr)
    print("direct access: m.atoms[0]:", m.atoms[0])

    print("looping m.bonds")
    for i, j, attr in m.bonds:
        print(i, j, attr)
    bond = (i, j)
    print("direct access: m.atoms[({},{})]:".format(i, j, m.bonds[(i, j)]))


def plot_molecules(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
    # filename="~/Applications/db_access/mol_builder/molecules_n200.pkl",
    plot_prefix="~/Applications/db_access/mol_builder",
):

    mols = pickle_load(filename)

    for m in mols:

        fname = os.path.join(
            plot_prefix,
            "mol_png/{}_{}_{}_{}.png".format(
                m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
            ),
        )
        # m.draw(fname, show_atom_idx=True)
        m.draw3(filename=fname, show_atom_idx=True)

        fname = expand_path(fname)
        subprocess.run(["convert", fname, "-trim", "-resize", "100%", fname])

        fname = os.path.join(
            plot_prefix,
            "mol_pdb/{}_{}_{}_{}.pdb".format(
                m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
            ),
        )
        m.write(fname, file_format="pdb")


def plot_atom_distance_hist(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
):
    """
    Plot the distance between atoms.
    """

    def plot_hist(data, filename):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(data, 20)

        ax.set_xlabel("Bond length")
        ax.set_ylabel("counts")

        fig.savefig(filename, bbox_inches="tight")

    def get_distances(m):
        dist = [
            np.linalg.norm(m.coords[u] - m.coords[v])
            for u, v in itertools.combinations(range(len(m.coords)), 2)
        ]
        return dist

    # prepare data
    mols = pickle_load(filename)
    data = [get_distances(m) for m in mols]
    data = np.concatenate(data)

    print("\n\n### atom distance min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/db_access/mol_builder/atom_distances.pdf"
    filename = expand_path(filename)
    plot_hist(data, filename)


def write_group_isomorphic_to_file():
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    filename = "~/Applications/db_access/mol_builder/isomorphic_mols.txt"
    DatabaseOperation.write_group_isomorphic_to_file(mols, filename)


def write_dataset():
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    # mols = mols[len(mols) * 739 // 2048 : len(mols) * 740 // 2048]

    # #######################
    # # filter charge 0 mols
    # #######################
    # new_mols = []
    # for m in mols:
    #     if m.charge == 1:
    #         new_mols.append(m)
    # mols = new_mols

    struct_file = "~/Applications/db_access/mol_builder/struct_mols.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_mols.csv"
    feature_file = "~/Applications/db_access/mol_builder/feature_mols.yaml"
    # struct_file = "~/Applications/db_access/mol_builder/struct_mols_charge1.sdf"
    # label_file = "~/Applications/db_access/mol_builder/label_mols_charge1.csv"
    # feature_file = "~/Applications/db_access/mol_builder/feature_mols_charge1.yaml"
    DatabaseOperation.write_sdf_csv_dataset(mols, struct_file, label_file, feature_file)


def detect_bad_mols():
    struct_file = "~/Applications/db_access/mol_builder/struct.sdf"
    struct_file = expand_path(struct_file)
    suppl = Chem.SDMolSupplier(struct_file, sanitize=True, removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            print("bad mol:", i)


def number_of_bonds():
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    mols = pickle_load(filename)

    nbonds = []
    for m in mols:
        nbonds.append(len(m.bonds))
    mean = np.mean(nbonds)
    median = np.median(nbonds)

    print("### number of bonds mean:", mean)
    print("### number of bonds median:", median)


def get_single_atom_energy():
    filename = "~/Applications/db_access/mol_builder/molecules_unfiltered.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    formula = ["H1", "Li1", "C1", "O1", "F1", "P1"]
    print("# formula    free energy    charge")
    for m in mols:
        if m.formula in formula:
            print(m.formula, m.free_energy, m.charge)


if __name__ == "__main__":
    # pickle_db_entries()
    # pickle_molecules()
    # print_mol_property()
    plot_molecules()
    # plot_atom_distance_hist()
    # number_of_bonds()

    # write_dataset()
    # detect_bad_mols()

    # write_group_isomorphic_to_file()
    # get_single_atom_energy()
