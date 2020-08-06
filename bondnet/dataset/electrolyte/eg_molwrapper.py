import numpy as np
from rdkit import Chem
from bondnet.core.molwrapper import write_edge_label_based_on_bond, write_sdf_csv_dataset
from bondnet.utils import pickle_load, to_path


def write_dataset():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
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

    struct_file = "~/Applications/db_access/mol_builder/struct_mols_n200.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_mols_n200.csv"
    feature_file = "~/Applications/db_access/mol_builder/feature_mols_n200.yaml"
    write_sdf_csv_dataset(mols, struct_file, label_file, feature_file)


def write_dataset_edge_label():
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    struct_file = "~/Applications/db_access/mol_builder/struct_mols_bond_annotation.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_mols_bond_annotation.yaml"
    feature_file = (
        "~/Applications/db_access/mol_builder/feature_mols_bond_annotation.yaml"
    )
    write_edge_label_based_on_bond(mols, struct_file, label_file, feature_file)


def detect_bad_mols():
    struct_file = "~/Applications/db_access/mol_builder/struct.sdf"
    struct_file = to_path(struct_file)
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

    write_dataset()
    # write_dataset_edge_label()
