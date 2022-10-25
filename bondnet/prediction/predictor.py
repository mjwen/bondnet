import numpy as np
import torch
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.prediction.io import (
    PredictionByReaction,
    PredictionMultiReactant,
    PredictionOneReactant,
    PredictionStructLabelFeatFiles,
)
from bondnet.prediction.load_model import (
    get_model_info,
    get_model_path,
    load_dataset,
    load_model,
)
from bondnet.utils import to_path


def predict_single_molecule(
    model_name,
    molecule,
    charge=0,
    ring_bond=False,
    one_per_iso_bond_group=True,
    write_result=False,
    figure_name="prediction.png",
    format=None,
):
    """
    Make predictions for a single molecule.

    Breaking a bond may result in products with different combination of products
    charge, we report the smallest charge w.r.t. the product charge assignation.

    Args:
        model_name (str): The pre-trained model to use for making predictions. A model
            should be of the format `dataset/date`, e.g. `bdncm/20200808`,
            `pubchem/20200521`. It is possible to provide only the `dataset` part,
            and in this case, the latest model will be used.
        molecule (str): SMILES or InChI string or a path to a file storing these string.
        charge (int): charge of the molecule.
        ring_bond (bool): whether to make predictions for ring bond.
        one_per_iso_bond_group (bool): If `True`, keep one reaction for each
            isomorphic bond group (fragments obtained by breaking different bond
            are isomorphic to each other). If `False`, keep all.
        write_result (bool): whether to write the returned sdf to stdout.
        figure_name (str): the name of the figure to be created showing the bond energy.
        format (str): format of the molecule, if not provided, will guess based on the
            file extension.

    Returns:
        str: sdf string representing the molecules and energies.
    """

    model_path = get_model_path(model_name)
    model_info = get_model_info(model_path)
    allowed_charge = model_info["allowed_charge"]
    unit_converter = model_info["unit_conversion"]

    assert (
        charge in allowed_charge
    ), f"expect charge to be one of {allowed_charge}, but got {charge}"

    p = to_path(molecule)
    if p.is_file():
        if format is None:
            suffix = p.suffix.lower()
            if suffix == ".sdf":
                format = "sdf"
            elif suffix == ".pdb":
                format = "pdb"
            else:
                raise RuntimeError(
                    f"Expect file format `.sdf` or `.pdb`, but got {suffix}"
                )
        with open(p, "r") as f:
            molecule = f.read().strip()
    else:
        if format is None:
            if molecule.lower().startswith("inchi="):
                format = "inchi"
            else:
                format = "smiles"

    predictor = PredictionOneReactant(
        molecule, charge, format, allowed_charge, ring_bond, one_per_iso_bond_group
    )

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(
        model_path, unit_converter, molecules, labels, extra_features
    )

    return predictor.write_results(predictions, figure_name, write_result)


def predict_multiple_molecules(
    model_name, molecule_file, charge_file, out_file, format
):
    """
    Make predictions of bond energies of multiple molecules.

    Args:
        model_name (str): The pretrained model to use for making predictions. A model
            should be of the format format `dataset/date`, e.g. `bdncm/20200808`,
            `pubchem/20200531`. It is possible to provide only the `dataset` part,
            and in this case, the latest model will be used.
        molecule_file (str): path to molecule file
        charge_file (str): path to charge file, if `None` charges are set to zero
        out_file (str): path to file to write output
        format (str): format of molecules, e.g. `sdf`, `graph`, `pdb`, `smiles`,
            and `inchi`.
    """

    model_path = get_model_path(model_name)
    model_info = get_model_info(model_path)
    allowed_charge = model_info["allowed_charge"]
    unit_converter = model_info["unit_conversion"]

    predictor = PredictionMultiReactant(
        molecule_file, charge_file, format, allowed_charge, ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(
        model_path, unit_converter, molecules, labels, extra_features
    )

    return predictor.write_results(predictions, out_file)


def predict_by_reactions(
    model_name, molecule_file, reaction_file, charge_file, out_file, format
):
    """
    Make predictions for many bonds where each bond is specified as an reaction.

    Args:
        model_name (str): The pretrained model to use for making predictions. A model
            should be of the format format `dataset/date`, e.g. `bdncm/20200808`,
            `pubchem/20200531`. It is possible to provide only the `dataset` part,
            and in this case, the latest model will be used.
        molecule_file (str): path to file storing all molecules
        reaction_file (str): path to file specifying reactions
        charge_file (str): path to charge file, if `None` charges are set to zero
        out_file (str): path to file to write output
        format (str): format of molecules, e.g. `sdf`, `graph`, `pdb`, `smiles`,
            and `inchi`.
    """
    model_path = get_model_path(model_name)
    model_info = get_model_info(model_path)
    unit_converter = model_info["unit_conversion"]

    predictor = PredictionByReaction(
        molecule_file, reaction_file, charge_file, format=format
    )

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(
        model_path, unit_converter, molecules, labels, extra_features
    )

    return predictor.write_results(predictions, out_file)


def predict_by_struct_label_extra_feats_files(
    model_name, molecule_file, label_file, extra_feats_file, out_file="bde.yaml"
):
    model_path = get_model_path(model_name)
    model_info = get_model_info(model_path)
    unit_converter = model_info["unit_conversion"]

    predictor = PredictionStructLabelFeatFiles(
        molecule_file, label_file, extra_feats_file
    )

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(
        model_path, unit_converter, molecules, labels, extra_features
    )

    return predictor.write_results(predictions, out_file)


def get_prediction(model_path, unit_converter, molecules, labels, extra_features):

    model = load_model(model_path)
    dataset = load_dataset(model_path, molecules, labels, extra_features)
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
                pred.append(predictions[idx] * unit_converter)
                idx += 1
        predictions = pred
    else:
        predictions = np.asarray(predictions) * unit_converter

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
