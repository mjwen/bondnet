"""
Write the bond features in the molecules for each GNN layer.

This is the code we used to generate data for LBDC bond similarity heatmap.
"""
import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.prediction.load_model import load_model, load_dataset
from gnn.utils import seed_torch


def evaluate(model, nodes, data_loader):
    model.eval()

    targets = []
    predictions = []
    ids = []
    features = defaultdict(list)

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

            fts = model.feature_at_each_layer(
                bg, feats, label["reaction"], norm_atom, norm_bond
            )

            for layer_idx, t in fts.items():
                # NOTE t[0] is the bond feature of the first molecule, i.e. the reactant
                features[layer_idx].append(t[0].numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)
    features = {layer_idx: np.concatenate(ft) for layer_idx, ft in features.items()}

    species = ["-".join(x.split("-")[-2:]) for x in ids]

    broken_bonds = []
    for x in ids:
        bonds = x.split("_")[2].split("-")[1:]
        broken_bonds.append(tuple(sorted([int(i) for i in bonds])))

    return ids, targets, predictions, broken_bonds, species, features


def main(
    model_name="electrolyte/20200528",
    sdf_file="/Users/mjwen/Applications/db_access/mol_builder/post_analysis/lbdc/struct.sdf",
    label_file="/Users/mjwen/Applications/db_access/mol_builder/post_analysis/lbdc/label.yaml",
    feature_file="/Users/mjwen/Applications/db_access/mol_builder/post_analysis/lbdc/feature.yaml",
    feat_meta_prefix=f"~/Applications/db_access/mol_builder/post_analysis/lbdc",
):

    seed_torch()

    dataset = load_dataset(model_name, sdf_file, label_file, feature_file)
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=100, shuffle=False)

    model = load_model(model_name, pretrained=False)

    # make predictions
    feature_names = ["atom", "bond", "global"]
    ids, targets, predictions, broken_bonds, species, features = evaluate(
        model, feature_names, data_loader
    )

    # write to file
    for idx, ft in features.items():
        fname = os.path.join(feat_meta_prefix, f"feats_layer{idx}.tsv")
        df = pd.DataFrame(ft)
        df.to_csv(fname, sep="\t", header=False, index=False)

    df = pd.DataFrame(
        {
            "identifier": ids,
            "target": targets,
            "prediction": predictions,
            "broken_bonds": broken_bonds,
            "species": species,
        }
    )
    fname = os.path.join(feat_meta_prefix, "feats_metadata.tsv")
    df.to_csv(fname, sep="\t", index=False)


if __name__ == "__main__":
    main()
