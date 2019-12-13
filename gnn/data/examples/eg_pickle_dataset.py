from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset


def get_dataset_electrolyte():
    return ElectrolyteDataset(
        sdf_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf",
        label_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt",
        self_loop=True,
        hetero=False,
        pickle_dataset=True,
    )


def get_pickled_electrolyte():
    return ElectrolyteDataset(
        sdf_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf.pkl",
        label_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt.pkl",
    )


def get_dataset_qm9():
    return QM9Dataset(
        sdf_file="/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf",
        label_file="/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf.csv",
        self_loop=True,
        hetero=False,
        pickle_dataset=True,
        properties=["u0_atom"],
        unit_conversion=True,
    )


def get_pickled_qm9():
    return QM9Dataset(
        sdf_file="/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf.pkl",
        label_file="/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf.csv.pkl",
    )


if __name__ == "__main__":
    # dataset = get_dataset_electrolyte()
    # dataset = get_pickled_electrolyte()

    dataset = get_dataset_qm9()
    # dataset = get_pickled_qm9()
