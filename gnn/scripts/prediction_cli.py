"""
Script to make predictions using command line interface.
"""

import os
import click

import gnn
from gnn.prediction.io import (
    PredictionOneReactant,
    PredictionMultiReactant,
    PredictionSDFChargeReactionFiles,
    PredictionMolGraphReactionFiles,
)
from gnn.prediction.prediction import get_prediction
from rdkit import RDLogger

# RDLogger.logger().setLevel(RDLogger.CRITICAL)

LATEST_NREL_MODEL = "20200422"
LATEST_ELECTROLYTE_MODEL = "20200528"

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=False)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["nrel", "electrolyte"], case_sensitive=False),
    default="electrolyte",
    show_default=True,
    help="prediction using model trained to the dataset",
)
@click.version_option(version=gnn.__version__)
@click.pass_context
def cli(ctx, model):
    model = model.lower()
    if model == "nrel":
        model = os.path.join(model, LATEST_NREL_MODEL)
        allowed_charge = [0]
    else:
        model = os.path.join(model, LATEST_ELECTROLYTE_MODEL)
        allowed_charge = [-1, 0, 1]

    model_info = {"model": model, "allowed_charge": allowed_charge}

    ctx.obj = model_info


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule", type=str)
@click.option("--charge", type=int, default=0, help="charge of the molecule")
@click.pass_obj
def single(model_info, molecule, charge):
    """
    Make predictions for a molecule.

    MOLECULE is a SMILES string or InChI string.
    """

    if molecule.lower().startswith("inchi="):
        format = "inchi"
    else:
        format = "smiles"
    predictor = PredictionOneReactant(
        molecule, charge, format, model_info["allowed_charge"], ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model_info["model"], molecules, labels, extra_features)
    predictor.write_results(predictions)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule-file", type=click.Path(exists=True))
@click.option(
    "-c",
    "--charge-file",
    type=click.Path(exists=True),
    help="charges of molecules; if not provided, charges set to 0",
)
@click.option(
    "-o",
    "--out-file",
    type=click.Path(exists=False),
    help="file to write results; if not provided, write to stdout",
)
@click.option(
    "-t",
    "--format",
    type=click.Choice(["sdf", "pdb", "smiles", "inchi"], case_sensitive=False),
    default="sdf",
    show_default=True,
    help="format of molecules",
)
@click.pass_obj
def multiple(model_info, molecule_file, charge_file, out_file, format):
    """
    Make predictions for multiple molecules.
    """

    predictor = PredictionMultiReactant(
        molecule_file, charge_file, format, model_info["allowed_charge"], ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model_info["model"], molecules, labels, extra_features)
    predictor.write_results(predictions, out_file)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule-file", type=click.Path(exists=True))
@click.argument("reaction-file", type=click.Path(exists=True))
@click.option(
    "-c",
    "--charge-file",
    type=click.Path(exists=True),
    help="charges of molecules; if not provided, charges set to 0",
)
@click.option(
    "-o",
    "--out-file",
    type=click.Path(exists=False),
    help="file to write results; if not provided, write to stdout",
)
@click.option(
    "-t",
    "--format",
    type=click.Choice(["sdf", "graph", "pdb", "smiles", "inchi"], case_sensitive=False),
    default="sdf",
    show_default=True,
    help="format of molecules",
)
@click.pass_obj
def reaction(model_info, molecule_file, reaction_file, charge_file, out_file, format):
    """
    Make predictions for bonds given as reactions.

    MOLECULE_FILE lists all the molecules.

    REACTION_FILE is a csv file lists bond breaking reactions the molecules can form,
    specified by the index of the molecules.
    """

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
    predictions = get_prediction(model_info["model"], molecules, labels, extra_features)
    predictor.write_results(predictions, out_file)
