from pathlib import Path
from click.testing import CliRunner
import bondnet
from bondnet.prediction.predictor import (
    predict_single_molecule,
    predict_multiple_molecules,
    predict_by_reactions,
)
from bondnet.scripts.predict_cli import cli


def test_predict_single_molecule():
    predict_single_molecule(model_name="bdncm", molecule="CC")
    predict_single_molecule(model_name="pubchem", molecule="CC")


def test_predict_multiple_molecules():
    prefix = Path(bondnet.__file__).parent.joinpath("scripts", "examples", "predict")
    molecule_file = prefix.joinpath("molecules.sdf")
    charge_file = prefix.joinpath("charges.txt")

    predict_multiple_molecules(
        model_name="bdncm",
        molecule_file=molecule_file,
        charge_file=charge_file,
        out_file="/tmp/bde.sdf",
        format="sdf",
    )

    predict_multiple_molecules(
        model_name="pubchem",
        molecule_file=molecule_file,
        charge_file=None,
        out_file="/tmp/bde.sdf",
        format="sdf",
    )


def test_predict_by_reaction():
    prefix = Path(bondnet.__file__).parent.joinpath("scripts", "examples", "predict")
    molecule_file = prefix.joinpath("molecules.sdf")
    rxn_file = prefix.joinpath("reactions.csv")
    charge_file = prefix.joinpath("charges.txt")

    predict_by_reactions(
        model_name="bdncm",
        molecule_file=molecule_file,
        reaction_file=rxn_file,
        charge_file=charge_file,
        out_file="/tmp/bde.csv",
        format="sdf",
    )

    predict_by_reactions(
        model_name="bdncm",
        molecule_file=molecule_file,
        reaction_file=rxn_file,
        charge_file=None,
        out_file="/tmp/bde.csv",
        format="sdf",
    )


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["single", "CC", "0"])

    molecule_file = Path(bondnet.__file__).parent.joinpath(
        "prediction", "examples", "molecules.sdf"
    )
    result = runner.invoke(cli, ["multiple", molecule_file])
