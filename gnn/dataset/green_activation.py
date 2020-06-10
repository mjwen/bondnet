import os
import logging
import multiprocessing
import pandas as pd
from gnn.core.rdmol import RdkitMolCreationError, smarts_atom_mapping, smarts_to_rdkit_mol
from gnn.core.molwrapper import rdkit_mol_to_wrapper_mol
from gnn.core.reaction import Reaction
from gnn.utils import expand_path


logger = logging.getLogger(__name__)


def smarts_to_wrapper_mol(s):
    try:
        m = smarts_to_rdkit_mol(s)
        m = rdkit_mol_to_wrapper_mol(m, charge=0, identifier=s)
    except RdkitMolCreationError:
        m = None
    return m


def read_dataset(filename):
    def get_idx(smiles, s):
        try:
            idx = smiles[s]
        except KeyError:
            idx = len(smiles)
            smiles[s] = idx
        return idx

    filename = expand_path(filename)
    df = pd.read_csv(filename, header=0, index_col=0)

    # remove duplicate reactions where reactant and products are the same
    selected_rxns = []
    rxn_set = set()
    for row in df.itertuples():
        rxn = (row[1], row[2])
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
        idx, reactant, product, activation_energy, enthalpy = rxn
        idx_r = get_idx(unique_smiles, reactant)
        idx_p = get_idx(unique_smiles, product)
        reactions_by_smiles_idx.append((idx, idx_r, idx_p, activation_energy, enthalpy))
        if i % 1000 == 0:
            logger.info(f"Finding unique smiles; processing {i}/{len(selected_rxns)}")

    logger.info(f"Unique molecules: {len(unique_smiles)}")

    # convert smiles to wrapper molecules
    smiles = sorted(unique_smiles, key=lambda k: unique_smiles[k])

    # molecules = [smarts_to_wrapper_mol(s) for s in smiles]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        molecules = p.map(smarts_to_wrapper_mol, smiles)

    # find unsuccessful conversion
    bad_mol_indices = {i for i, m in enumerate(molecules) if m is None}
    if bad_mol_indices:
        bad_ones = ""
        for idx in bad_mol_indices:
            bad_ones += f"{smiles[idx]}, "
        logger.warning(f"Bad mol; (rdkit conversion failed): {bad_ones}")

    # convert to reactions
    reactions = []
    for idx, idx_r, idx_p, act_e, eth_e in reactions_by_smiles_idx:
        for i in (idx_r, idx_p):
            if i in bad_mol_indices:
                logger.warning(
                    "Ignore bad reaction (conversion its mol failed): "
                    "{} -> {}. Bad mol is: {}".format(
                        smiles[idx_r], smiles[idx_p], smiles[i],
                    )
                )
                break
        else:
            rxn = Reaction(
                reactants=[molecules[idx_r]],
                products=[molecules[idx_p]],
                broken_bond=None,
                free_energy=act_e,
                identifier=idx,
            )
            mp = get_atom_mapping(
                smarts_atom_mapping(smiles[idx_r]), smarts_atom_mapping(smiles[idx_p])
            )
            rxn.set_atom_mapping([mp])

            reactions.append(rxn)

    logger.info(f"Finish converting {len(reactions)} reactions")

    return reactions


def get_atom_mapping(amp1, amp2):
    """
    Map atom index in amp2 to index in amp1.

    Args:
        amp1 (list):
        amp2 (list):

    Returns:
        dict: {idx2:idx1} where idx2 is atom index in mol 2 and idx1 is atom idx in mol 1
    """
    assert len(amp1) == len(amp2), "amp1 and amp2 have different length"

    mapping = {}
    for idx2, atom in enumerate(amp2):
        idx1 = amp1.index(atom)
        mapping[idx2] = idx1

    return mapping


def plot_molecules(filename, plot_prefix):
    reactions = read_dataset(filename)

    for rxn in reactions:
        molecules = rxn.reactants + rxn.products
        rxn_id = rxn.get_id()

        for i, m in enumerate(molecules):
            r_or_p = "rct" if i == 0 else "prdt"

            fname = os.path.join(
                plot_prefix,
                "mol_png/{}_{}_{}_{}.png".format(rxn_id, r_or_p, m.formula, m.charge),
            )
            m.draw(fname, show_atom_idx=True)

            fname = os.path.join(
                plot_prefix,
                "mol_pdb/{}_{}_{}_{}.pdb".format(rxn_id, r_or_p, m.formula, m.charge),
            )
            m.write(fname, format="pdb")


if __name__ == "__main__":
    # filename = "~/Documents/Dataset/activation_energy_Green/wb97xd3_n200.csv"
    # plot_prefix = "~/Applications/db_access/greens_rxns"
    # plot_molecules(filename, plot_prefix)

    filename = "~/Documents/Dataset/activation_energy_Green/wb97xd3_n200.csv"
    reactions = read_dataset(filename)
