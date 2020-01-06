from gnn.data.database import DatabaseOperation


def eg_query_database():
    db = DatabaseOperation.from_query()
    db.to_file("~/Applications/mongo_db_access/extracted_data/database.pkl")


def test_get_job_types():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_data/job_types.yml"
    db.get_job_types(filename)


def test_select(n=200):
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_data/database_n{}.pkl".format(n)
    db.to_file(filename, size=n)


def test_filter():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    # db.filter(keys=["formula_pretty"], value="LiH4(CO)3")
    # db.to_file(filename="~/Applications/mongo_db_access/extracted_data/database_LiEC.pkl")

    db.filter(keys=["formula_alphabetical"], values=["H1", "H2"])
    db.to_file(filename="~/Applications/mongo_db_access/extracted_data/database_H.pkl")


def test_molecules():
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


def test_write_group_isomorphic_to_file():
    db_path = "~/Applications/mongo_db_access/extracted_data/database.pkl"
    # db_path = "~/Applications/mongo_db_access/extracted_data/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True, purify=True)
    filename = "/Users/mjwen/Applications/mongo_db_access/extracted_data/isomorphic.txt"
    db.write_group_isomorphic_to_file(mols, filename)


if __name__ == "__main__":
    # eg_query_database()
    # test_select()
    # test_get_job_types()
    # test_filter()
    # test_create_dataset()
    # test_molecules()
    # test_write_group_isomorphic_to_file()

    eg_plot_charge_0()
