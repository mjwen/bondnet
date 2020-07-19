import os
from click.testing import CliRunner
import bondnet
from bondnet.prediction.predictor import (
    predict_single_molecule,
    predict_multiple_molecules,
    predict_by_reactions,
)
from bondnet.scripts.prediction_cli import cli


def test_predict_single_molecule():
    predict_single_molecule(model="electrolyte", molecule="CC")
    predict_single_molecule(model="alfabet", molecule="CC")


def test_predict_multiple_molecules():
    prefix = os.path.join(os.path.dirname(bondnet.__file__), "prediction", "examples")
    molecule_file = os.path.join(prefix, "molecules.sdf")
    charge_file = os.path.join(prefix, "charges.txt")

    predict_multiple_molecules(
        model="electrolyte",
        molecule_file=molecule_file,
        charge_file=charge_file,
        out_file="/tmp/bde.sdf",
        format="sdf",
    )

    predict_multiple_molecules(
        model="alfabet",
        molecule_file=molecule_file,
        charge_file=None,
        out_file="/tmp/bde.sdf",
        format="sdf",
    )


def test_predict_by_reaction():
    prefix = os.path.join(os.path.dirname(bondnet.__file__), "prediction", "examples")
    molecule_file = os.path.join(prefix, "molecules.sdf")
    rxn_file = os.path.join(prefix, "reactions.csv")
    charge_file = os.path.join(prefix, "charges.txt")

    predict_by_reactions(
        model="electrolyte",
        molecule_file=molecule_file,
        reaction_file=rxn_file,
        charge_file=charge_file,
        out_file="/tmp/bde.csv",
        format="sdf",
    )

    predict_by_reactions(
        model="electrolyte",
        molecule_file=molecule_file,
        reaction_file=rxn_file,
        charge_file=None,
        out_file="/tmp/bde.csv",
        format="sdf",
    )


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["single", "CC", "0"])

    molecule_file = os.path.join(
        os.path.dirname(bondnet.__file__), "prediction", "examples", "molecules.sdf"
    )
    result = runner.invoke(cli, ["multiple", molecule_file])
