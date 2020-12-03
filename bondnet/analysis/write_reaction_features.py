"""
Write the final features representing the reactions before the FC layers.

This is the code we used to write features and metadata for UMAP embedding.
"""

import torch
import numpy as np
import pandas as pd
from bondnet.data.dataset import train_validation_test_split
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.prediction.load_model import load_model, load_dataset
from bondnet.utils import seed_torch, to_path, yaml_load


def get_charges(label_file, feature_file):
    """
    Charge of reactant and products molecule in each reaction.
    """
    labels = yaml_load(label_file)
    features = yaml_load(feature_file)

    ids = []
    num_prdts = []
    rct_charges = []
    prdt1_charges = []
    prdt2_charges = []

    for lb in labels:
        ids.append(lb["id"])
        rct_idx = lb["reactants"][0]
        prdts = lb["products"]

        N = len(prdts)
        num_prdts.append(N)

        rct_charges.append(features[rct_idx]["charge"])
        prdt1_idx = prdts[0]
        prdt1_charges.append(features[prdt1_idx]["charge"])
        if N == 2:
            prdt2_idx = prdts[1]
            prdt2_charges.append(features[prdt2_idx]["charge"])
        else:
            prdt2_charges.append(None)

    df = pd.DataFrame(
        {
            "identifier": ids,
            "num products": num_prdts,
            "charge": rct_charges,
            "product1 charge": prdt1_charges,
            "product2 charge": prdt2_charges,
        }
    )
    return df


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

    if compute_features:
        features = np.concatenate(features)
    else:
        features = None

    return ids, targets, predictions, errors, species, features


def main(
    model_name="bdncm/20200808",
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
    df.to_csv(to_path(feat_filename), sep="\t", header=False, index=False)

    # metadata
    charges = get_charges(label_file, feature_file)
    rct_charges = []
    prdt1_charges = []
    prdt2_charges = []
    for i in ids:
        c = charges[charges["identifier"] == i].to_dict("records")[0]
        rct_charges.append(c["charge"])
        prdt1_charges.append(c["product1 charge"])
        prdt2_charges.append(c["product2 charge"])

    df = pd.DataFrame(
        {
            "identifier": ids,
            "target": targets,
            "prediction": predictions,
            "error": errors,
            "species": species,
            "reactant charge": rct_charges,
            "product1 charge": prdt1_charges,
            "product2 charge": prdt2_charges,
        }
    )
    df.to_csv(to_path(meta_filename), sep="\t", index=False)


if __name__ == "__main__":
    main()
