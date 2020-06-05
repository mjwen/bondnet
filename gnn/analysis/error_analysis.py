import torch
import numpy as np
import pandas as pd
from gnn.data.dataset import train_validation_test_split
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.prediction.load_model import load_model, load_dataset
from gnn.utils import seed_torch, expand_path, yaml_load


def evaluate(model, nodes, data_loader):
    model.eval()

    targets = []
    predictions = []
    ids = []

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

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)
    species = ["-".join(x.split("-")[-2:]) for x in ids]
    errors = predictions - targets

    space_removed_ids = [
        s.replace(", ", "-").replace("(", "").replace(")", "") for s in ids
    ]

    return space_removed_ids, targets, predictions, errors, species


def get_charges(label_file, feature_file):
    """
    Charge of reactant molecule in each reaction.
    """
    labels = yaml_load(expand_path(label_file))
    features = yaml_load(expand_path(feature_file))
    ids = []
    charges = []

    for lb in labels:
        ids.append(lb["id"])
        reactant_idx = lb["reactants"][0]
        charges.append(features[reactant_idx]["charge"])

    return ids, charges


def main(
    model_name="electrolyte/20200528",
    sdf_file="~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_qc.sdf",
    label_file="~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_qc.yaml",
    feature_file="~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_qc.yaml",
    error_file="~/Applications/db_access/mol_builder/post_analysis/evaluation_error.tsv",
    charge_file="~/Applications/db_access/mol_builder/post_analysis/charges.tsv",
):

    seed_torch()

    dataset = load_dataset(model_name, sdf_file, label_file, feature_file)
    _, _, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)
    data_loader = DataLoaderReactionNetwork(testset, batch_size=100, shuffle=False)
    # data_loader = DataLoaderReactionNetwork(dataset, batch_size=100, shuffle=False)

    model = load_model(model_name)

    # make predictions
    feature_names = ["atom", "bond", "global"]
    ids, targets, predictions, errors, species = evaluate(
        model, feature_names, data_loader
    )

    # sort by error
    ids, targets, predictions, errors, species = zip(
        *sorted(zip(ids, targets, predictions, errors, species), key=lambda x: x[3])
    )

    df = pd.DataFrame(
        {
            "identifier": ids,
            "target": targets,
            "prediction": predictions,
            "error": errors,
            "species": species,
        }
    )

    df.to_csv(expand_path(error_file), sep="\t", index=False)

    # charges
    ids, charges = get_charges(label_file, feature_file)
    df = pd.DataFrame({"identifier": ids, "charge": charges})
    df.to_csv(expand_path(charge_file), sep="\t", index=False)


if __name__ == "__main__":
    main()
