import pytest
from gnn.data.query_db import DatabaseOperation

# def test_query_database():
#     db = DatabaseOperation.from_query()
#     db.to_file(db.entries)


def test_filter():

    db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    db.filter(keys=['formula_pretty'], value='LiH4(CO)3')
    db.to_file(
        filename='/Users/mjwen/Applications/mongo_db_access/extracted_data/database_LiEC.pkl'
    )


def test_create_dataset():

    db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules(purify=True)
    # mols = mols[:6]
    db.create_sdf_csv_dataset(
        mols,
        '/Users/mjwen/Applications/mongo_db_access/extracted_data/electrolyte_LiEC.sdf',
        '/Users/mjwen/Applications/mongo_db_access/extracted_data/electrolyte_LiEC.csv',
    )
    print('entries saved:', len(mols))


def test_molecules():
    db_path = '/Users/mjwen/Applications/mongo_db_access/extracted_data/database_LiEC.pkl'
    db = DatabaseOperation.from_file(db_path)
    mols = db.to_molecules()
    for i, m in enumerate(mols):
        print(i, m.id)
        m.draw('./images/mol_{}.svg'.format(i))


if __name__ == '__main__':
    # test_filter()
    # test_create_dataset()
    test_molecules()
