"""
Write the final features representing the reactions before the FC layers.
"""

import torch
import numpy as np
import pandas as pd
from gnn.data.dataset import train_validation_test_split
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.prediction.load_model import load_model, load_dataset
from gnn.utils import seed_torch, expand_path


def evaluate(model, nodes, data_loader, compute_features=False):
    model.eval()

    targets = []
    predictions = []
    ids = []
    features = []

    with torch.no_grad():
        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            tgt = label["value"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            mean = label["scaler_mean"]
            stdev = label["scaler_stdev"]

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1) * stdev + mean

            tgt = tgt * stdev + mean

            predictions.append(pred.numpy())
            targets.append(tgt.numpy())
            ids.append([rxn.id for rxn in label["reaction"]])

            if compute_features:
                feats = model.feature_before_fc(
                    bg, feats, label["reaction"], norm_atom, norm_bond
                )
                features.append(feats.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)
    species = ["-".join(x.split("-")[-2:]) for x in ids]
    errors = predictions - targets

    space_removed_ids = [
        s.replace(", ", "-").replace("(", "").replace(")", "") for s in ids
    ]

    if compute_features:
        features = np.concatenate(features)
    else:
        features = None

    return space_removed_ids, targets, predictions, errors, species, features


def main(
    model_name="electrolyte/20200528",
    sdf_file="~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_qc.sdf",
    label_file="~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_qc.yaml",
    feature_file="~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_qc.yaml",
    feat_filename="~/Applications/db_access/mol_builder/post_analysis/feats.tsv",
    meta_filename="~/Applications/db_access/mol_builder/post_analysis/feats_metadata.tsv",
):

    seed_torch()

    dataset = load_dataset(model_name, sdf_file, label_file, feature_file)
    _, _, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)
    data_loader = DataLoaderReactionNetwork(testset, batch_size=100, shuffle=False)
    # data_loader = DataLoaderReactionNetwork(dataset, batch_size=100, shuffle=False)

    model = load_model(model_name)

    # make predictions
    feature_names = ["atom", "bond", "global"]
    ids, targets, predictions, errors, species, features = evaluate(
        model, feature_names, data_loader, compute_features=True
    )

    df = pd.DataFrame(features)
    df.to_csv(expand_path(feat_filename), sep="\t", header=False, index=False)
    df = pd.DataFrame(
        {
            "identifier": ids,
            "target": targets,
            "prediction": predictions,
            "error": errors,
            "species": species,
        }
    )
    df.to_csv(expand_path(meta_filename), sep="\t", index=False)


if __name__ == "__main__":
    main()
