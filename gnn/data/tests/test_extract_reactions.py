import pytest
from gnn.data.query_db import DatabaseOperation, ReactionExtractor
from gnn.data.utils import print_dict


def test_buckets():

    # test get isomer
    db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database_LiEC.pkl'
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()

    extractor = ReactionExtractor(molecules)
    buckets = extractor.bucket_molecules(keys=['formula', 'charge', 'spin_multiplicity'])
    print_dict(buckets)
    buckets = extractor.bucket_molecules(keys=['formula'])
    print_dict(buckets)


def test_extract_A_to_B():
    # db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database_LiEC.pkl'
    db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()
    print('db recovered, number of mols:', len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=['formula', 'charge'])
    print('number of buckets', len(extractor.buckets))

    extractor.extract_A_to_B_style_reaction()
    extractor.to_file(
        filename='/Users/mjwen/Applications/mongo_db_access/extracted_data/A2B_rxns.pkl'
    )


def test_load_A_to_B():
    filename = '/Users/mjwen/Applications/mongo_db_access/extracted_data/A2B_rxns.pkl'
    extractor = ReactionExtractor.from_file(filename)
    print('Number of reactions', len(extractor.reactions))


def test_extract_A_to_B_C():
    # db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database_LiEC.pkl'
    db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    molecules = db.to_molecules()
    print('db recovered, number of moles:', len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=['formula', 'charge'])
    print('number of buckets', len(extractor.buckets))

    extractor.extract_A_to_B_C_style_reaction()
    extractor.to_file(
        filename='/Users/mjwen/Applications/mongo_db_access/extracted_data/A2BC_rxns.json'
    )


if __name__ == '__main__':
    # test_buckets()
    # test_extract_A_to_B()
    # test_extract_A_to_B_C()
    test_load_A_to_B()
