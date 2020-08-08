import torch
import numpy as np
import pandas as pd
from bondnet.data.dataset import (
    train_validation_test_split,
    train_validation_test_split_selected_bond_in_train,
)
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.prediction.load_model import load_model, load_dataset
from bondnet.utils import seed_torch, to_path, yaml_load


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


def main(
    model_name="mesd/20200808",
    sdf_file="~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_qc.sdf",
    label_file="~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_qc.yaml",
    feature_file="~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_qc.yaml",
    error_file="~/Applications/db_access/mol_builder/post_analysis/evaluation_error.tsv",
    charge_file="~/Applications/db_access/mol_builder/post_analysis/charges.tsv",
):

    seed_torch()

    dataset = load_dataset(model_name, sdf_file, label_file, feature_file)

    # trainset, valset, testset = train_validation_test_split(
    #     dataset, validation=0.1, test=0.1
    # )

    trainset, valset, testset = train_validation_test_split_selected_bond_in_train(
        dataset,
        validation=0.1,
        test=0.1,
        selected_bond_type=(("H", "H"), ("H", "F"), ("F", "F")),
    )

    # data_loader = DataLoaderReactionNetwork(trainset, batch_size=100, shuffle=False)
    # data_loader = DataLoaderReactionNetwork(valset, batch_size=100, shuffle=False)
    data_loader = DataLoaderReactionNetwork(testset, batch_size=100, shuffle=False)

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

    df.to_csv(to_path(error_file), sep="\t", index=False)

    # charges
    df = get_charges(label_file, feature_file)
    df.to_csv(to_path(charge_file), sep="\t", index=False)


if __name__ == "__main__":
    main()
