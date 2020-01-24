from gnn.data.database_molecules import DatabaseOperation
from gnn.utils import pickle_dump, pickle_load, yaml_dump, expand_path


def eg_query_database():
    # db = DatabaseOperation.from_query()
    # db.to_file("~/Applications/mongo_db_access/extracted_mols/database.pkl")
    db = DatabaseOperation.from_query(num_entries=200)
    db.to_file("~/Applications/mongo_db_access/extracted_mols/database_n200.pkl")


def eg_select(n=200):
    db_path = "~/Applications/mongo_db_access/extracted_mols/database.pkl"
    db = DatabaseOperation.from_file(db_path)
    filename = "~/Applications/mongo_db_access/extracted_mols/database_n{}.pkl".format(n)
    db.to_file(filename, size=n)


def pickle_molecules():
    # db_path = "~/Applications/mongo_db_access/extracted_mols/database.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_mols/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules()

    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    pickle_dump(mols, filename)


def eg_molecules():
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    mols = pickle_load(filename)

    for m in mols:
        # fname = "~/Applications/mongo_db_access/extracted_mols/mol_svg/{}_{}_{}_{}.svg".format(
        # m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
        # )
        fname = "~/Applications/mongo_db_access/extracted_mols/mol_png/{}_{}_{}_{}.png".format(
            m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
        )
        m.draw(fname, show_atom_idx=True)

        fname = "/Users/mjwen/Applications/mongo_db_access/extracted_mols/pdb/{}_{}_{}_{}.pdb".format(
            m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
        )
        m.write(fname, file_format="pdb")


def write_features():
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    mols = pickle_load(filename)

    all_feats = dict()
    for m in mols:
        feat = m.pack_features(use_obabel_idx=True)
        all_feats[feat["id"]] = feat

    filename = "~/Applications/mongo_db_access/extracted_mols/features_n200.yaml"
    yaml_dump(all_feats, filename)


def eg_write_group_isomorphic_to_file():
    db_path = "~/Applications/mongo_db_access/extracted_mols/database.pkl"
    # db_path = "~/Applications/mongo_db_access/extracted_mols/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(optimized=True, purify=True)
    filename = "/Users/mjwen/Applications/mongo_db_access/extracted_mols/isomorphic.txt"
    db.write_group_isomorphic_to_file(mols, filename)


def write_dataset():
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    mols = pickle_load(filename)

    #######################
    # filter charge 0 mols
    #######################
    new_mols = []
    for m in mols:
        if m.charge == 0:
            new_mols.append(m)
    mols = new_mols

    # structure_name = "~/Applications/mongo_db_access/extracted_mols/struct_mols.sdf"
    # label_name = "~/Applications/mongo_db_access/extracted_mols/label_mols.csv"
    structure_name = (
        "~/Applications/mongo_db_access/extracted_mols/struct_mols_charge0.sdf"
    )
    label_name = "~/Applications/mongo_db_access/extracted_mols/label_mols_charge0.csv"
    DatabaseOperation.write_sdf_csv_dataset(mols, structure_name, label_name)
    DatabaseOperation.write_sdf_csv_dataset(mols, structure_name, label_name)


def get_single_atom_energy():
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    mols = pickle_load(filename)

    formula = ["H1", "Li1", "C1", "O1", "F1", "P1"]
    print("# formula    free energy    charge")
    for m in mols:
        if m.formula in formula:
            print(m.formula, m.free_energy, m.charge)


if __name__ == "__main__":
    eg_query_database()
    # eg_select()
    pickle_molecules()
    # eg_molecules()
    write_features()

    # write_dataset()

    # get_single_atom_energy()
    # eg_write_group_isomorphic_to_file()
