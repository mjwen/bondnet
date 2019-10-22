import pytest
from gnn.data.query_db import DatabaseOperation

# def test_query_database():
#     db = DatabaseOperation.from_query()
#     db.to_file(db.entries)


def test_filter():

    db_path = '/Users/mjwen/Applications/mongo_db_access/database.pkl'
    db = DatabaseOperation.from_file(db_path)
    LiEC_entries = db.filter(db.entries, keys=['formula_pretty'], value='LiH4(CO)3')
    db.to_file(
        LiEC_entries,
        size=None,
        filename='/Users/mjwen/Applications/mongo_db_access/database_LiEC.pkl',
    )


def test_create_dataset():

    db_path = '/Users/mjwen/Applications/mongo_db_access/database_LiEC.pkl'
    db = DatabaseOperation.from_file(db_path)
    entries = db.entries
    # entries = db.entries[:6]
    db.create_sdf_csv_dataset(entries, 'electrolyte_LiEC.sdf', 'electrolyte_LiEC.csv')
    print('entries saved:', len(entries))


if __name__ == '__main__':
    # test_filter()
    test_create_dataset()
