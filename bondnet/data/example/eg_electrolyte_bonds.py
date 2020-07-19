from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizerWithReactionInfo,
    BondAsNodeFeaturizer,
    GlobalFeaturizerWithReactionInfo,
)
from bondnet.data.electrolyte import ElectrolyteBondDataset


def get_dataset(
    struct_file="~/Applications/db_access/mol_builder/struct_clfn_qc_ws.sdf",
    label_file="~/Applications/db_access/mol_builder/label_clfn_qc_ws.txt",
    feature_file="~/Applications/db_access/mol_builder/feature_clfn_qc_ws.yaml",
):
    """
    By running this, we observe the output to get a sense of the low and high values
    for bond length featurizer.

    """
    grapher = HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizerWithReactionInfo(),
        bond_featurizer=BondAsNodeFeaturizer(
            # length_featurizer="bin",
            # length_featurizer_args={"low": 0.7, "high": 2.5, "num_bins": 10},
            length_featurizer="rbf",
            length_featurizer_args={"low": 0.2, "high": 2.7, "num_centers": 20},
        ),
        global_featurizer=GlobalFeaturizerWithReactionInfo(),
        self_loop=True,
    )

    dataset = ElectrolyteBondDataset(
        grapher=grapher,
        sdf_file=struct_file,
        label_file=label_file,
        feature_file=feature_file,
    )

    return dataset


if __name__ == "__main__":
    get_dataset()
