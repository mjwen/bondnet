from gnn.data.database import DatabaseOperation
from gnn.utils import pickle_dump, pickle_load


def pickle_molecules():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    # db_path = "~/Applications/mongo_db_access/extracted_data/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True)

    filename = "~/Applications/mongo_db_access/extracted_data/molecules.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_data/molecules_n200.pkl"
    pickle_dump(mols, filename)


def eg_query_database():
    db = DatabaseOperation.from_query()
    db.to_file("~/Applications/mongo_db_access/extracted_data/database.pkl")


def eg_get_job_types():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_data/job_types.yml"
    db.get_job_types(filename)


def eg_select(n=200):
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_data/database_n{}.pkl".format(n)
    db.to_file(filename, size=n)


def eg_filter():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    # db.filter(keys=["formula_pretty"], value="LiH4(CO)3")
    # db.to_file(filename="~/Applications/mongo_db_access/extracted_data/database_LiEC.pkl")

    db.filter(keys=["formula_alphabetical"], values=["H1", "H2"])
    db.to_file(filename="~/Applications/mongo_db_access/extracted_data/database_H.pkl")


def eg_molecules():
    # db_path = "~/Applications/mongo_db_access/extracted_data/database_LiEC.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True)
    for m in mols:
        fname = "~/Applications/mongo_db_access/extracted_data/mol_svg/{}_{}.svg".format(
            m.formula, m.id
        )
        m.draw(fname, show_atom_idx=True)

        fname = "/Users/mjwen/Applications/mongo_db_access/extracted_data/pdb/{}_{}.pdb".format(
            m.formula, m.id
        )
        m.write(fname, file_format="pdb")


def eg_plot_charge_0():
    # db_path = "~/Applications/mongo_db_access/extracted_data/database_LiEC.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True)
    for m in mols:
        if m.charge == 0:

            fname = "~/Applications/mongo_db_access/extracted_data/mol_svg/{}_{}.svg".format(
                m.formula, m.id
            )
            m.draw(fname, show_atom_idx=False)


def eg_write_group_isomorphic_to_file():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    # db_path = "~/Applications/mongo_db_access/extracted_data/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True, purify=True)
    filename = "/Users/mjwen/Applications/mongo_db_access/extracted_data/isomorphic.txt"
    db.write_group_isomorphic_to_file(mols, filename)


def write_dataset():
    filename = "~/Applications/mongo_db_access/extracted_data/molecules.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_data/molecules_n200.pkl"
    mols = pickle_load(filename)

    #######################
    # filter charge 0 mols
    #######################
    new_mols = []
    for m in mols:
        if m.charge == 0:
            new_mols.append(m)
    mols = new_mols

    # structure_name = "~/Applications/mongo_db_access/extracted_data/struct_mols.sdf"
    # label_name = "~/Applications/mongo_db_access/extracted_data/label_mols.csv"
    structure_name = (
        "~/Applications/mongo_db_access/extracted_data/struct_mols_charge0.sdf"
    )
    label_name = "~/Applications/mongo_db_access/extracted_data/label_mols_charge0.csv"
    DatabaseOperation.write_sdf_csv_dataset(mols, structure_name, label_name)
    DatabaseOperation.write_sdf_csv_dataset(mols, structure_name, label_name)


def get_single_atom_energy():
    filename = "~/Applications/mongo_db_access/extracted_data/molecules.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_data/molecules_n200.pkl"
    mols = pickle_load(filename)

    formula = ["H1", "Li1", "C1", "O1", "F1", "P1"]
    print("# formula    free energy    charge")
    for m in mols:
        if m.formula in formula:
            print(m.formula, m.free_energy, m.charge)


if __name__ == "__main__":
    # eg_query_database()
    # eg_select()
    # eg_get_job_types()
    # eg_filter()
    # eg_create_dataset()
    # eg_molecules()
    # eg_write_group_isomorphic_to_file()
    # eg_plot_charge_0()

    # pickle_molecules()

    write_dataset()

    # get_single_atom_energy()
