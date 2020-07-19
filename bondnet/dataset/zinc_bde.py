import os
import glob
import logging
from rdkit import Chem
from bondnet.core.molwrapper import rdkit_mol_to_wrapper_mol
from bondnet.core.reaction import ReactionExtractorFromReactant
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import expand_path


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
    for i, fname in enumerate(filenames):
        m = Chem.MolFromMolFile(fname, sanitize=True, removeHs=False)
        if m is None:
            n_bad += 1
        else:
            title, energies = parse_title_and_energies(fname)
            mw = rdkit_mol_to_wrapper_mol(m, charge=0, identifier=f"{title}_{i}")
            mols.append(mw)
            bond_energies.append(energies)
    print(f"{n_bad} bad molecules ignored.")

    mol_reservoir = set(mols)

    n_bad = 0
    reactions = []
    for m, e in zip(mols, bond_energies):
        extractor = ReactionExtractorFromReactant(m, e, allowed_charge=[0])
        extractor.extract(ring_bond=True, mol_reservoir=mol_reservoir)
        reactions.extend(extractor.reactions)
        for bond, reason in extractor.no_reaction_reason.items():
            if reason.compute and reason.fail:
                # print(
                #     f"Creating reaction by breaking bond {bond} of molecule {m.id} "
                #     f"fails because {reason.reason}"
                # )
                n_bad += 1

    print(f"{n_bad} bond cannot be broken to get products.")

    return reactions


def plot_zinc_mols(dirname="~/Documents/Dataset/ZINC_BDE"):

    dirname = expand_path(dirname)

    filenames = glob.glob(os.path.join(dirname, "*.sdf"))
    for i, fname in enumerate(filenames):
        m = Chem.MolFromMolFile(fname, sanitize=True, removeHs=False)
        if m is not None:
            m = rdkit_mol_to_wrapper_mol(m)
            identifier = os.path.splitext(os.path.basename(fname))[0]
            prefix = "~/Applications/db_access/zinc_bde/mol_png"
            m.draw(os.path.join(prefix, identifier + ".png"), show_atom_idx=True)

            prefix = "~/Applications/db_access/zinc_bde/mol_pdb"
            m.write(os.path.join(prefix, identifier + ".pdb"), format="pdb")


def zinc_create_struct_label_dataset_bond_based_regression(
    dirname="~/Documents/Dataset/ZINC_BDE_100",
    # dirname="~/Documents/Dataset/ZINC_BDE",
):
    reactions = read_zinc_bde_dataset(dirname)
    extractor = ReactionCollection(reactions)

    extractor.create_struct_label_dataset_bond_based_regression(
        struct_file="~/Applications/db_access/zinc_bde/zinc_struct_bond_rgrn.sdf",
        label_file="~/Applications/db_access/zinc_bde/zinc_label_bond_rgrn.txt",
        feature_file="~/Applications/db_access/zinc_bde/zinc_feature_bond_rgrn.yaml",
        # struct_file="~/Applications/db_access/zinc_bde/zinc_struct_bond_rgrn_n200.sdf",
        # label_file="~/Applications/db_access/zinc_bde/zinc_label_bond_rgrn_n200.txt",
        # feature_file="~/Applications/db_access/zinc_bde/zinc_feature_bond_rgrn_n200.yaml",
        group_mode="charge_0",
        one_per_iso_bond_group=True,
    )


def zinc_create_struct_label_dataset_reaction_network_based_regression(
    # dirname="~/Documents/Dataset/ZINC_BDE_100",
    dirname="~/Documents/Dataset/ZINC_BDE",
):

    reactions = read_zinc_bde_dataset(dirname)
    extractor = ReactionCollection(reactions)

    extractor.create_regression_dataset_reaction_network_simple(
        struct_file="~/Applications/db_access/zinc_bde/zinc_struct_rxn_ntwk_rgrn.sdf",
        label_file="~/Applications/db_access/zinc_bde/zinc_label_rxn_ntwk_rgrn.yaml",
        feature_file="~/Applications/db_access/zinc_bde/zinc_feature_rxn_ntwk_rgrn.yaml",
        # struct_file="~/Applications/db_access/zinc_bde/zinc_struct_rxn_ntwk_rgrn_n200.sdf",
        # label_file="~/Applications/db_access/zinc_bde/zinc_label_rxn_ntwk_rgrn_n200.yaml",
        # feature_file="~/Applications/db_access/zinc_bde/zinc_feature_rxn_ntwk_rgrn_n200.yaml",
    )


if __name__ == "__main__":

    # plot_zinc_mols()
    # zinc_create_struct_label_dataset_bond_based_regression()
    zinc_create_struct_label_dataset_reaction_network_based_regression()
