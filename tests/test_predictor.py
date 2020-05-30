import os
from click.testing import CliRunner
import gnn
from gnn.scripts.prediction_cli import cli


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["single", "CC", "0"])

    molecule_file = os.path.join(
        os.path.dirname(gnn.__file__), "prediction", "examples", "molecules.sdf"
    )
    result = runner.invoke(cli, ["multiple", molecule_file])
