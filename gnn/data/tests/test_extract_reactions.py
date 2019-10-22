import pytest
import itertools
import numpy as np

from gnn.data.query_db import (
    DatabaseOperation,
    Molecule,
    ReactionExtractor,
    load_extracted_reactions,
)
from gnn.data.utils import print_dict


def test_buckets():

    # test get isomer
    db_path = '/Users/mjwen/Applications/mongo_db_access/database_LiEC.pkl'
    db = DatabaseOperation.from_file(db_path)
    molecules = DatabaseOperation.to_molecules(db.entries)

    extractor = ReactionExtractor(molecules)
    buckets = extractor.bucket_molecules(keys=['formula', 'charge', 'spin_multiplicity'])
    print_dict(buckets)
    buckets = extractor.bucket_molecules(keys=['formula'])
    print_dict(buckets)


def test_extract_A_to_B():
    # db_path = '/Users/mjwen/Applications/mongo_db_access/database_LiEC.pkl'
    db_path = '/Users/mjwen/Applications/mongo_db_access/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    molecules = DatabaseOperation.to_molecules(db.entries)
    print('db recovered, number of moles:', len(molecules))

    extractor = ReactionExtractor(molecules)
    buckets = extractor.bucket_molecules(keys=['formula', 'charge'])
    print('number of buckets', len(buckets))

    A2B_rxns = extractor.extract_A_to_B_style_reaction()
    extractor.to_file(A2B_rxns, filename='A2B_rxns.json')


def test_extract_A_to_B_C():
    # db_path = '/Users/mjwen/Applications/mongo_db_access/database_LiEC.pkl'
    db_path = '/Users/mjwen/Applications/mongo_db_access/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    molecules = DatabaseOperation.to_molecules(db.entries)
    print('db recovered, number of moles:', len(molecules))

    extractor = ReactionExtractor(molecules)
    buckets = extractor.bucket_molecules(keys=['formula', 'charge'])
    print('number of buckets', len(buckets))

    A2BC_rxns = extractor.extract_A_to_B_C_style_reaction()
    extractor.to_file(A2BC_rxns, filename='A2BC_rxns.json')



if __name__ == '__main__':
    # test_buckets()
    # test_extract_A_to_B()
    # test_extract_A_to_B_C()
