import os
import torch
import numpy as np
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.prediction.io import (
    PredictionOneReactant,
    PredictionMultiReactant,
    PredictionSDFChargeReactionFiles,
    PredictionMolGraphReactionFiles,
)
from gnn.prediction.load_model import load_model, load_dataset
from gnn.utils import expand_path
from rdkit import RDLogger

# RDLogger.logger().setLevel(RDLogger.CRITICAL)

MODEL_INFO = {
    "electrolyte": {"allowed_charge": [-1, 0, 1], "date": ["20200422", "20200528"]},
    "nrel": {"allowed_charge": [-1, 0, 1], "date": ["20200422"]},
}


def predict_single_molecule(
    model,
    molecule,
    charge=0,
    ring_bond=False,
    write_result=False,
    figure_name="prediction.png",
    format=None,
):
    """
    Make predictions for a single molecule.

    Args:
        model (str): The pretrained model to use for making predictions. A model should
            be of the format format `dataset/date`, e.g. `electrolyte/20200528`,
            `nrel/20200528`. It is possible to provide only the `dataset` part,
            and in this case, the latest model will be used.
        molecule (str): SMILES string or InChI string.
        charge (int): charge of the molecule.
        ring_bond (bool): whether to make predictions for ring bond.
        write_result (bool): whether to write the returned sdf to stdout.
        figure_name (str): the name of the figure to be created showing the bond energy.
        format (str): format of the molecule, if not provided, will guess based on the
            file extension.

    Returns:
        str: sdf string representing the molecules and energies.
    """

    model, allowed_charge = select_model(model)

    assert (
        charge in allowed_charge
    ), f"expect charge to be one of {allowed_charge}, but got {charge}"

    if os.path.isfile(molecule):
        if format is None:
            extension = os.path.splitext(molecule)[1]
            if extension.lower() == ".sdf":
                format = "sdf"
            elif extension.lower() == ".pdb":
                format = "pdb"
            else:
                raise RuntimeError(
                    f"Expect file format `.sdf` or `.pdb`, but got {extension}"
                )
        with open(expand_path(molecule), "r") as f:
            molecule = f.read().strip()
    else:
        if format is None:
            if molecule.lower().startswith("inchi="):
                format = "inchi"
            else:
                format = "smiles"

    predictor = PredictionOneReactant(molecule, charge, format, allowed_charge, ring_bond)

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model, molecules, labels, extra_features)

    return predictor.write_results(predictions, figure_name, write_result)


def predict_multiple_molecules(model, molecule_file, charge_file, out_file, format):

    model, allowed_charge = select_model(model)

    predictor = PredictionMultiReactant(
        molecule_file, charge_file, format, allowed_charge, ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model, molecules, labels, extra_features)
    return predictor.write_results(predictions, out_file)


def predict_by_reactions(
    model, molecule_file, reaction_file, charge_file, out_file, format
):

    model, allowed_charge = select_model(model)

    # sdf 3 files: mol (in sdf), charge (in plain text), reaction (csv)
    if format == "sdf":
        predictor = PredictionSDFChargeReactionFiles(
            molecule_file, charge_file, reaction_file
        )

    # mol graph 2 files, mol (json or yaml), reaction (csv)
    elif format == "graph":
        predictor = PredictionMolGraphReactionFiles(molecule_file, reaction_file)

    # # internal 3 files: sdf file, label file, feature file
    # elif format == "internal":
    #     predictor = PredictionStructLabelFeatFiles(molecule_file, label_file, feat_file)

    else:
        raise ValueError(f"not supported molecule format: {format}")

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model, molecules, labels, extra_features)
    return predictor.write_results(predictions, out_file)


def select_model(model_name):
    """
    Select the appropriate model.

    A `model_name` can be provided in two format:
    1. `dataset/data`, in this case the model will be returned directly.
    2. `dataset`, in this case the latest model corresponds to the dataset is returned.

    Args:
        model_name (str)

    Returns:
        str: the model to use
    """
    model_name = model_name.strip().lower()
    if "/" in model_name:
        prefix, date = model_name.split("/")
        if date not in MODEL_INFO[prefix]["date"]:
            raise ValueError(
                f"expect model date to be one of { MODEL_INFO[prefix]['date'] }, "
                f"but got {date}."
            )
    else:
        prefix = model_name
        date = MODEL_INFO[prefix]["date"][-1]

    model = "/".join([prefix, date])
    allowed_charge = MODEL_INFO[prefix]["allowed_charge"]

    return model, allowed_charge


def get_prediction(model_name, molecules, labels, extra_features):

    model = load_model(model_name)
    dataset = load_dataset(model_name, molecules, labels, extra_features)
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=100, shuffle=False)

    feature_names = ["atom", "bond", "global"]

    # evaluate
    predictions = evaluate(model, feature_names, data_loader)

    # in case some entry fail
    if len(predictions) != len(dataset.failed):
        pred = []
        idx = 0
        for failed in dataset.failed:
            if failed:
                pred.append(None)
            else:
                pred.append(predictions[idx])
                idx += 1
        predictions = pred

    return predictions


def evaluate(model, nodes, data_loader, device=None):
    model.eval()

    predictions = []
    with torch.no_grad():

        for it, (bg, label) in enumerate(data_loader):
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            mean = label["scaler_mean"]
            stdev = label["scaler_stdev"]

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1)
            pred = (pred * stdev + mean).cpu().numpy()

            predictions.append(pred)

    predictions = np.concatenate(predictions)

    return predictions
