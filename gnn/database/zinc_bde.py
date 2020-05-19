import os
import glob
import logging
from rdkit import Chem
from gnn.database.molwrapper import rdkit_mol_to_wrapper_mol
from gnn.utils import expand_path


logger = logging.getLogger(__name__)


def read_zinc_bde_dataset(dirname):
    """
    Read the zinc bde dataset.

    The dataset is described in: Qu et al. Journal of Cheminformatics 2013, 5:34
    which can be obtained from: https://doi.org/10.1186/1758-2946-5-34
    See the `Electronic supplementary material` section.

    Args:
        dirname (str): directory name contains the sdf files.

    Returns:
        mols (list): a sequence of :class:`MoleculeWrapper`.
        energies (list of dict): bond energies. Each dict for one molecule, with bond
            index (a tuple) as key and bond energy as value.
    """

    def parse_title_and_energies(filename):
        """
        Returns:
            title (str): first line of sdf file.
            energies (dict): with bond index (a tuple) as key and bond energy as value.

        """
        with open(filename, "r") as f:
            lines = f.readlines()

        title = lines[0].strip()

        energies = dict()
        for ln in lines:
            ln = ln.strip().split()
            if len(ln) == 8:
                # -1 to convert to 0 based
                bond = tuple(sorted([int(ln[0]) - 1, int(ln[1]) - 1]))
                e = float(ln[7])
                energies[bond] = e

        return title, energies

    dirname = expand_path(dirname)
    if not os.path.isdir(dirname):
        raise ValueError(f"expect dirname to be a directory, but got {dirname}")

    n_bad = 0
    mols = []
    bond_energies = []

    filenames = glob.glob(os.path.join(dirname, "*.sdf"))
    for fname in filenames:
        m = Chem.MolFromMolFile(fname, sanitize=True, removeHs=False)
        if m is None:
            n_bad += 1
            logger.warning(f"bad mol: {fname}")
        else:
            title, energies = parse_title_and_energies(fname)
            mw = rdkit_mol_to_wrapper_mol(m, charge=0, identifier=title)
            mols.append(mw)
            bond_energies.append(energies)
    logger.warning(f"{n_bad} bad mols ignored.")

    return mols, bond_energies


if __name__ == "__main__":

    read_zinc_bde_dataset("~/Documents/Dataset/ZINC_BDE_100")
