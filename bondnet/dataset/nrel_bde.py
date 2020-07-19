import os
import logging
import multiprocessing
import pandas as pd
from bondnet.core.rdmol import smiles_to_rdkit_mol, RdkitMolCreationError
from bondnet.core.molwrapper import rdkit_mol_to_wrapper_mol
from bondnet.core.reaction import Reaction
from bondnet.core.reaction_collection import (
    ReactionCollection,
    get_molecules_from_reactions,
)
from bondnet.utils import expand_path


logger = logging.getLogger(__name__)


def smiles_to_wrapper_mol(s):
    try:
        m = smiles_to_rdkit_mol(s)
        m = rdkit_mol_to_wrapper_mol(m, charge=0, identifier=s)
    except RdkitMolCreationError:
        m = None
    return m


def read_nrel_bde_dataset(filename):
    """
    Read the zinc bde dataset.

    The dataset is described in:
    https://chemrxiv.org/articles/Prediction_of_Homolytic_Bond_Dissociation_Enthalpies_for_Organic_Molecules_at_near_Chemical_Accuracy_with_Sub-Second_Computational_Cost/10052048/2

    Args:
        filename (str): csv file containing the bde info

    Returns:
        list: a sequence of :class:`bondnet.database.reaction.Reaction`
    """

    def get_idx(smiles, s):
        try:
            idx = smiles[s]
        except KeyError:
            idx = len(smiles)
            smiles[s] = idx
        return idx

    filename = expand_path(filename)
    df = pd.read_csv(filename, header=0, index_col=None)

    # remove duplicate reactions where reactant and products are the same
    selected_rxns = []
    rxn_set = set()
    for row in df.itertuples():
        rxn = (row[2], tuple(sorted((row[4], row[5]))))
        if rxn not in rxn_set:
            rxn_set.add(rxn)
            selected_rxns.append(row)

    logger.info(f"Number of reactions: {df.shape[0]}")
    logger.info(f"Duplicate reactions: {df.shape[0] - len(selected_rxns)}")
    logger.info(f"Remaining reactions: {len(selected_rxns)}")

    # find a unique smiles and represent reactions by index in it
    unique_smiles = {}  # smiles as key, index as value
    reactions_by_smiles_idx = []
    for i, rxn in enumerate(selected_rxns):
        idx, rid, reactant, bond_index, product1, product2, bde, bond_type = rxn
        idx_r = get_idx(unique_smiles, reactant)
        idx_p1 = get_idx(unique_smiles, product1)
        idx_p2 = get_idx(unique_smiles, product2)
        reactions_by_smiles_idx.append((idx_r, idx_p1, idx_p2, rid, bde))
        if i % 1000 == 0:
            logger.info(f"Finding unique smiles; processing {i}/{len(selected_rxns)}")

    logger.info(f"Total number of molecules: {3*len(reactions_by_smiles_idx)}")
    logger.info(f"Unique molecules: {len(unique_smiles)}")

    # convert smiles to wrapper molecules
    smiles = sorted(unique_smiles, key=lambda k: unique_smiles[k])

    # molecules = [smiles_to_wrapper_mol(s) for s in smiles]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        molecules = p.map(smiles_to_wrapper_mol, smiles)

    # find unsuccessful conversion
    bad_mol_indices = {i for i, m in enumerate(molecules) if m is None}
    if bad_mol_indices:
        bad_ones = ""
        for idx in bad_mol_indices:
            bad_ones += f"{smiles[idx]}, "
        logger.warning(f"Bad mol; (rdkit conversion failed): {bad_ones}")

    # convert to reactions
    reactions = []
    for idx_r, idx_p1, idx_p2, rid, bde in reactions_by_smiles_idx:
        for i in (idx_r, idx_p1, idx_p2):
            if i in bad_mol_indices:
                logger.warning(
                    "Ignore bad reaction (conversion its mol failed): "
                    "{} -> {} + {}. Bad mol is: {}".format(
                        smiles[idx_r], smiles[idx_p1], smiles[idx_p2], smiles[i],
                    )
                )
                break
        else:
            reactions.append(
                Reaction(
                    reactants=[molecules[idx_r]],
                    products=[molecules[idx_p1], molecules[idx_p2]],
                    broken_bond=None,
                    free_energy=bde,
                    identifier=rid,
                )
            )

    logger.info(f"Finish converting {len(reactions)} reactions")

    return reactions


def nrel_plot_molecules(
    filename="~/Documents/Dataset/NREL_BDE/rdf_data_190531_n200.csv",
    plot_prefix="~/Applications/db_access/nrel_bde",
):
    reactions = read_nrel_bde_dataset(filename)
    molecules = get_molecules_from_reactions(reactions)

    unique_mols = {m.id: m for m in molecules}
    molecules = list(unique_mols.keys())

    for m in molecules:

        fname = os.path.join(
            plot_prefix,
            "mol_png/{}_{}_{}_{}.png".format(
                m.id, m.formula, m.charge, str(m.free_energy).replace(".", "dot")
            ),
        )
        m.draw(fname, show_atom_idx=True)

        fname = os.path.join(
            plot_prefix,
            "mol_pdb/{}_{}_{}_{}.pdb".format(
                m.id, m.formula, m.charge, str(m.free_energy).replace(".", "dot")
            ),
        )
        m.write(fname, format="pdb")


def nrel_create_struct_label_dataset_reaction_network_based_regression(
    filename="~/Documents/Dataset/NREL_BDE/rdf_data_190531_n200.csv",
):
    reactions = read_nrel_bde_dataset(filename)
    extractor = ReactionCollection(reactions)

    extractor.create_regression_dataset_reaction_network_simple(
        struct_file="~/Applications/db_access/nrel_bde/nrel_struct_rxn_ntwk_rgrn_n200.sdf",
        label_file="~/Applications/db_access/nrel_bde/nrel_label_rxn_ntwk_rgrn_n200.yaml",
        feature_file="~/Applications/db_access/nrel_bde/nrel_feature_rxn_ntwk_rgrn_n200.yaml",
    )


def nrel_create_struct_label_dataset_bond_based_regression(
    filename="~/Documents/Dataset/NREL_BDE/rdf_data_190531_n200.csv",
):
    reactions = read_nrel_bde_dataset(filename)
    extractor = ReactionCollection(reactions)

    extractor.create_struct_label_dataset_bond_based_regression(
        struct_file="~/Applications/db_access/nrel_bde/nrel_struct_bond_rgrn_n200.sdf",
        label_file="~/Applications/db_access/nrel_bde/nrel_label_bond_rgrn_n200.yaml",
        feature_file="~/Applications/db_access/nrel_bde/nrel_feature_bond_rgrn_n200.yaml",
    )


if __name__ == "__main__":
    # nrel_plot_molecules()
    # nrel_create_struct_label_dataset_reaction_network_based_regression()
    nrel_create_struct_label_dataset_bond_based_regression()
