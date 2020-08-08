import numpy as np
from rdkit import Chem
from bondnet.dataset.electrolyte.db_molecule import DatabaseOperation
from bondnet.dataset.electrolyte.db_molecule_quality_analysis import (
    check_bond_length,
    check_bond_species,
    check_connectivity,
    check_rdkit_sanitize,
    remove_mols_containing_species,
    plot_molecules,
    get_single_atom_molecule_energy,
)
from bondnet.utils import pickle_dump, pickle_load, to_path


def pickle_db_entries(filename="~/Applications/db_access/mol_builder/database_n200.pkl"):
    entries = DatabaseOperation.query_db_entries(
        db_collection="mol_builder", num_entries=200
    )
    # entries = DatabaseOperation.query_db_entries(db_collection="smd", num_entries=200)

    pickle_dump(entries, filename)


def pickle_molecules(outname, num_entries=500, db_file=None):

    db_collection = "mol_builder"
    # db_collection = "task"
    entries = DatabaseOperation.query_db_entries(
        db_collection=db_collection, db_file=db_file, num_entries=num_entries,
    )

    mols = DatabaseOperation.to_molecules(entries, db_collection=db_collection)

    # filename = "~/Applications/db_access/mol_builder/molecules_n200_unfiltered.pkl"
    # pickle_dump(mols, filename)

    mols = DatabaseOperation.filter_molecules(mols, connectivity=True, isomorphism=True)
    pickle_dump(mols, outname)


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

    print("\n\nlooping m.bonds")
    for bond, attr in m.bonds.items():
        print(bond, attr)


def write_group_isomorphic_to_file():
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    filename = "~/Applications/db_access/mol_builder/isomorphic_mols.txt"
    DatabaseOperation.write_group_isomorphic_to_file(mols, filename)


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


def check_all(filename, output_prefix=None):
    filename = to_path(filename)

    mols = pickle_load(filename)
    print("Number of mols before any check:", len(mols))

    if output_prefix is None:
        output_prefix = filename.parent

    mols = check_connectivity(
        mols=mols,
        metal="Li",
        filename_failed=output_prefix.joinpath("failed_connectivity.pkl"),
    )
    mols = check_rdkit_sanitize(
        mols=mols, filename_failed=output_prefix.joinpath("failed_rdkit_sanitize.pkl")
    )
    mols = check_bond_species(
        mols=mols, filename_failed=output_prefix.joinpath("failed_bond_species.pkl")
    )
    mols = check_bond_length(
        mols=mols, filename_failed=output_prefix.joinpath("failed_bond length.pkl")
    )
    mols = remove_mols_containing_species(
        mols=mols,
        species=["P"],
        filename_failed=output_prefix.joinpath("failed_containing_species.pkl"),
    )

    print("Number of mols after check:", len(mols))

    outname = output_prefix.joinpath(filename.stem + "_qc" + filename.suffix)
    pickle_dump(mols, outname)


if __name__ == "__main__":

    # # pickle_db_entries()
    # pickle_molecules(
    #     outname="~/Applications/db_access/mol_builder/molecules.pkl", num_entries=None
    # )
    #
    # check_all(filename="~/applications/db_access/mol_builder/molecules.pkl")

    get_single_atom_molecule_energy("~/applications/db_access/mol_builder/molecules.pkl")

    # print_mol_property()

    # plot_molecules(
    # filename = "~/Applications/db_access/mol_builder/molecules_qc.pkl",
    # plot_prefix = "~/Applications/db_access/mol_builder",
    # )

    # plot_atom_distance_hist()

    # number_of_bonds()
    # detect_bad_mols()

    # write_group_isomorphic_to_file()
