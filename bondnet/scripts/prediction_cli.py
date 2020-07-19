"""
Script to make predictions using command line interface.
"""

import click
import bondnet
from bondnet.prediction.predictor import (
    predict_single_molecule,
    predict_multiple_molecules,
    predict_by_reactions,
)
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=False)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["alfabet", "electrolyte"], case_sensitive=False),
    default="electrolyte",
    show_default=True,
    help="prediction using model trained to the dataset",
)
@click.version_option(version=bondnet.__version__)
@click.pass_context
def cli(ctx, model):
    ctx.obj = model


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule", type=str)
@click.option("--charge", type=int, default=0, help="charge of the molecule")
@click.option(
    "--ring-bond/--no-ring-bond",
    default=False,
    help="make prediction for bonds in a ring",
)
@click.pass_obj
def single(model, molecule, charge, ring_bond):
    """
    Make predictions for a molecule.

    MOLECULE is a SMILES string or InChI string.
    """
    return predict_single_molecule(model, molecule, charge, ring_bond, write_result=True)


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
def multiple(model, molecule_file, charge_file, out_file, format):
    """
    Make predictions for multiple molecules.
    """
    return predict_multiple_molecules(model, molecule_file, charge_file, out_file, format)


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
def reaction(model, molecule_file, reaction_file, charge_file, out_file, format):
    """
    Make predictions for bonds given as reactions.

    MOLECULE_FILE lists all the molecules.

    REACTION_FILE is a csv file lists bond breaking reactions the molecules can form,
    specified by the index of the molecules.
    """
    return predict_by_reactions(
        model, molecule_file, reaction_file, charge_file, out_file, format
    )
