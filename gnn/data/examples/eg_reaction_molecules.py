from gnn.data.database_molecules import DatabaseOperation
from gnn.data.reaction import ReactionExtractor
from pprint import pprint


def eg_buckets():

    # test get isomer
    db_path = "~/Applications/mongo_db_access/extracted_mols/database_LiEC.pkl"
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge", "spin_multiplicity"])
    pprint(extractor.buckets)
    extractor.bucket_molecules(keys=["formula"])
    pprint(extractor.buckets)


def eg_extract_A_to_B():
    # db_path = '~/Applications/mongo_db_access/extracted_mols/database_LiEC.pkl'
    # db_path = "~/Applications/mongo_db_access/extracted_mols/database.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_mols/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()
    print("db recovered, number of mols:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_A_to_B_style_reaction()
    extractor.to_file(
        filename="~/Applications/mongo_db_access/extracted_mols/reaction_A2B.pkl"
    )


def eg_extract_A_to_B_C():
    # db_path = '~/Applications/mongo_db_access/extracted_mols/database_LiEC.pkl'
    # db_path = "~/Applications/mongo_db_access/extracted_mols/database.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_mols/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()
    print("db recovered, number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_A_to_B_C_style_reaction()
    extractor.to_file(
        filename="~/Applications/mongo_db_access/extracted_mols/reactions_A2BC.pkl"
    )


def eg_extract_one_bond_break():
    # db_path = "~/Applications/mongo_db_access/extracted_mols/database.pkl"
    db_path = "~/Applications/mongo_db_access/extracted_mols/database_n200.pkl"
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()
    print("db recovered, number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_one_bond_break()

    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    extractor.to_file(filename)


def eg_reactants_bond_energies_to_file():
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    extractor = ReactionExtractor.from_file(filename)

    ##############
    # filter reactant charge = 0
    ##############
    extractor.filter_reactions_by_reactant_charge(charge=0)

    ##############
    # filter C-C bond and order
    ##############
    extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"), bond_order=1)

    filename = "~/Applications/mongo_db_access/extracted_mols/bond_energies.yaml"
    # filename = "~/Applications/mongo_db_access/extracted_mols/bond_energies_n200.yaml"
    extractor.bond_energies_to_file(filename)


def eg_create_struct_label_dataset():
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_A2B.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_A2BC.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    extractor = ReactionExtractor.from_file(filename)
    extractor.create_struct_label_dataset(
        struct_name="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
        label_name="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
    )


def eg_create_struct_label_dataset_with_lowest_energy_across_charge():
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_charge0.pkl"
    extractor = ReactionExtractor.from_file(filename)
    extractor.create_struct_label_dataset_with_lowest_energy_across_charge(
        struct_name="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
        label_name="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
        # struct_name="~/Applications/mongo_db_access/extracted_mols/struct_charge0.sdf",
        # label_name="~/Applications/mongo_db_access/extracted_mols/label_charge0.txt",
    )


def eg_create_struct_label_dataset_with_lowest_energy_across_charge_bond_based():
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    extractor = ReactionExtractor.from_file(filename)

    # ##############
    # # filter reactant charge = 0
    # ##############
    # extractor.filter_reactions_by_reactant_charge(charge=0)
    #
    # ##############
    # # filter C-C bond
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"), bond_order=1)

    extractor.create_struct_label_dataset_with_lowest_energy_across_charge_bond_based(
        # struct_name="~/Applications/mongo_db_access/extracted_mols/struct.sdf",
        # label_name="~/Applications/mongo_db_access/extracted_mols/label.txt",
        struct_name="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
        label_name="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
        feature_name="~/Applications/mongo_db_access/extracted_mols/feature_n200.json",
        # struct_name="~/Applications/mongo_db_access/extracted_mols/struct_charge0.sdf",
        # label_name="~/Applications/mongo_db_access/extracted_mols/label_charge0.txt",
        # struct_name="~/Applications/mongo_db_access/extracted_mols/struct_charge0_CC.sdf",
        # label_name="~/Applications/mongo_db_access/extracted_mols/label_charge0_CC.txt",
    )


if __name__ == "__main__":
    # eg_buckets()
    # eg_extract_A_to_B()
    # eg_extract_A_to_B_C()
    # eg_extract_one_bond_break()
    # eg_reactants_bond_energies_to_file()
    # eg_create_struct_label_dataset()
    # eg_create_struct_label_dataset_with_lowest_energy_across_charge()
    eg_create_struct_label_dataset_with_lowest_energy_across_charge_bond_based()
