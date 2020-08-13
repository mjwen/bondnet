from bondnet.core.molecule_collection import MoleculeCollection
from bondnet.dataset.electrolyte.db_molecule import DatabaseOperation
from bondnet.utils import pickle_dump, to_path


def pickle_db_entries(filename="~/Applications/db_access/mol_builder/database_n200.pkl"):
    entries = DatabaseOperation.query_db_entries(
        db_collection="mol_builder", num_entries=200
    )
    # entries = DatabaseOperation.query_db_entries(db_collection="smd", num_entries=200)

    pickle_dump(entries, filename)


def pickle_molecules(outname, num_entries=500, db_collection="mol_builder", db_file=None):

    if db_file is None:
        if db_collection == "mol_builder":
            db_file = "/Users/mjwen/Applications/db_access/sam_db/sam_db_mol_builder.json"
        elif db_collection == "task":
            db_file = "/Users/mjwen/Applications/db_access/sam_db/sam_db_tasks.json"
        else:
            raise Exception("Unrecognized db_collection = {}".format(db_collection))

    entries = DatabaseOperation.query_db_entries(
        db_collection=db_collection, db_file=db_file, num_entries=num_entries,
    )

    mols = DatabaseOperation.to_molecules(entries, db_collection=db_collection)

    # filename = "~/Applications/db_access/mol_builder/molecules_n200_unfiltered.pkl"
    # pickle_dump(mols, filename)

    mols = DatabaseOperation.filter_molecules(mols, connectivity=True, isomorphism=True)
    pickle_dump(mols, outname)


def check_all(filename="molecules.pkl", outname="molecules_qc.pkl"):

    mol_coll = MoleculeCollection.from_file(filename)

    print("Number of mols before any check:", len(mol_coll))

    mol_coll.filter_by_connectivity(exclude_species=["Li"])
    print("Number of mols after connectivity check:", len(mol_coll))

    mol_coll.filter_by_rdkit_sanitize()
    print("Number of mols after rdkit check:", len(mol_coll))

    mol_coll.filter_by_bond_species()
    print("Number of mols after bond species check:", len(mol_coll))

    mol_coll.filter_by_bond_length()
    print("Number of mols after bond length check:", len(mol_coll))

    mol_coll.filter_by_species(species=["P"])
    print("Number of mols after species check:", len(mol_coll))

    mol_coll.to_file(to_path(outname))


if __name__ == "__main__":

    working_dir = to_path("~/Applications/db_access/mol_builder/")

    # num_entries = 500
    # filename = working_dir.joinpath("molecules_n200.pkl")
    # # # num_entries = None
    # # filename = working_dir.joinpath(molecules.pkl")
    # pickle_molecules(num_entries=num_entries, outname=filename)

    filename = working_dir.joinpath("molecules_n200.pkl")
    outname = working_dir.joinpath("molecules_n200_qc.pkl")
    # filename = working_dir.joinpath("molecules.pkl")
    # outname = working_dir.joinpath("molecules_qc.pkl")
    check_all(filename, outname)

    # filename = working_dir.joinpath("molecules_qc.pkl")
    # mol_coll = MoleculeCollection.from_file(filename)
    # print(mol_coll.get_species())

    # filename = working_dir.joinpath("molecules_qc.pkl")
    # mol_coll = MoleculeCollection.from_file(filename)
    # print(mol_coll.get_molecule_counts_by_charge())

    # filename = working_dir.joinpath("molecules_qc.pkl")
    # mol_coll = MoleculeCollection.from_file(filename)
    # mol_coll.print_single_atom_property()

    # filename = working_dir.joinpath("molecules_qc.pkl")
    # mol_coll = MoleculeCollection.from_file(filename)
    # mol_coll.plot_molecules(prefix="~/Applications/db_mg")
