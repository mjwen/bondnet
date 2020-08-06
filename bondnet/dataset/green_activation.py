import logging
import multiprocessing
import warnings
import pandas as pd
from collections import defaultdict
from bondnet.core.rdmol import (
    RdkitMolCreationError,
    smarts_atom_mapping,
    smarts_to_rdkit_mol,
)
from bondnet.core.molwrapper import rdkit_mol_to_wrapper_mol
from bondnet.core.reaction import Reaction
from bondnet.utils import to_path


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

    filename = to_path(filename)
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

    plot_prefix = to_path(plot_prefix)

    for rxn in reactions:
        molecules = rxn.reactants + rxn.products
        rxn_id = rxn.get_id()

        for i, m in enumerate(molecules):
            r_or_p = "rct" if i == 0 else "prdt"

            fname = plot_prefix.joinpath(
                "mol_png/{}_{}_{}_{}.png".format(rxn_id, r_or_p, m.formula, m.charge),
            )
            # since the molecules have AtomMapNum set when reading smarts mol,
            # we set `show_atom_idx` to False to use the AtomMapNum there
            m.draw(fname, show_atom_idx=False)

            fname = plot_prefix.joinpath(
                "mol_pdb/{}_{}_{}_{}.pdb".format(rxn_id, r_or_p, m.formula, m.charge),
            )
            m.write(fname, format="pdb")


def bucket_rxns_by_num_altered_bonds(reactions):
    """
    Bucket the reactions dependning on the number altered bonds between reactants and
    products.

    Args:
        reactions (list): a sequence of core.reaction.Reaction objects

    Returns:
        dict: {num_altered_bonds:reactions} where reactions is a list of reactions
    """
    rxns_bucket = defaultdict(list)

    for rxn in reactions:
        prdt_to_rct_mapping = rxn.atom_mapping()[0]  # 0 because only one product
        rct_to_prdt_mapping = {v: k for k, v in prdt_to_rct_mapping.items()}
        rct_bonds = list(rxn.reactants[0].bonds.keys())
        prdt_bonds = list(rxn.products[0].bonds.keys())

        # reactant bond not in product
        n = 0
        for bond in rct_bonds:
            bond = tuple(sorted([rct_to_prdt_mapping[i] for i in bond]))
            if bond not in prdt_bonds:
                n += 1
        for bond in prdt_bonds:
            bond = tuple(sorted([prdt_to_rct_mapping[i] for i in bond]))
            if bond not in rct_bonds:
                n += 1

        rxns_bucket[n].append(rxn)

    num_rxns = {k: len(v) for k, v in rxns_bucket.items()}
    print(f"number of bond changes: {num_rxns}")

    return rxns_bucket


def bucket_rxns_by_altered_bond_types(reactions, n_bonds_altered=1):
    """
    Returns:
        dict: {(reactant_bond_not_product, product_bond_not_in_reactant):reactions}
    """
    rxns_bucket = defaultdict(list)

    for rxn in reactions:

        reactant = rxn.reactants[0]
        product = rxn.products[0]

        prdt_to_rct_mapping = rxn.atom_mapping()[0]  # 0 because only one product
        rct_to_prdt_mapping = {v: k for k, v in prdt_to_rct_mapping.items()}
        rct_bonds = list(reactant.bonds.keys())
        prdt_bonds = list(product.bonds.keys())

        rct_bonds_not_in_prdt = []
        prdt_bonds_not_in_rct = []
        for bond in rct_bonds:
            b = tuple(sorted([rct_to_prdt_mapping[i] for i in bond]))
            if b not in prdt_bonds:
                rct_bonds_not_in_prdt.append(bond)
        for bond in prdt_bonds:
            b = tuple(sorted([prdt_to_rct_mapping[i] for i in bond]))
            if b not in rct_bonds:
                prdt_bonds_not_in_rct.append(bond)

        # NOTE, a temporary check to make sure we only have bond breaking rxn
        if n_bonds_altered == 1:
            if prdt_bonds_not_in_rct:
                warnings.warn(f"prdt bonds altered for rxn: {rxn.get_id()}")

        rct_bonds_not_in_prdt_type = tuple(
            sorted(
                [
                    tuple(sorted([reactant.species[i] for i in bond]))
                    for bond in rct_bonds_not_in_prdt
                ]
            )
        )
        prdt_bonds_not_in_rct_type = tuple(
            sorted(
                [
                    tuple(sorted([product.species[i] for i in bond]))
                    for bond in prdt_bonds_not_in_rct
                ]
            )
        )

        rxns_bucket[(rct_bonds_not_in_prdt_type, prdt_bonds_not_in_rct_type)].append(rxn)

    num_rxns = {k: len(v) for k, v in rxns_bucket.items()}
    print(f"type of bonds changes: {num_rxns}")

    return rxns_bucket


def select_one_bond_break_reactions(filename, outname="one_bond_break.csv"):

    filename = to_path(filename)
    df = pd.read_csv(filename, header=0, index_col=0)

    reactions = read_dataset(filename)
    num_altered_bucket = bucket_rxns_by_num_altered_bonds(reactions)
    altered_type_bucket = bucket_rxns_by_altered_bond_types(num_altered_bucket[1], 1)

    rxn_id = []
    activation_e = []
    enthalpy = []
    broken_bond = []
    breaking = []

    for (rct_bond, prdt_bond), reactions in altered_type_bucket.items():
        for rxn in reactions:

            act_e = df["ea"][rxn.get_id()]
            eth_e = df["dh"][rxn.get_id()]

            # one bond breaking
            if len(rct_bond) == 1 and len(prdt_bond) == 0:
                bond = rct_bond[0]
                brk = True
            # one bond formation, we make it like bond breaking
            elif len(rct_bond) == 0 and len(prdt_bond) == 1:
                bond = prdt_bond[0]
                brk = False
            else:
                raise RuntimeError("not one bond alternation reaction encountered")

            rxn_id.append(rxn.get_id())
            activation_e.append(act_e)
            enthalpy.append(eth_e)
            broken_bond.append("-".join(bond))
            breaking.append(brk)

    df = pd.DataFrame(
        {
            "id": rxn_id,
            "bond type": broken_bond,
            "activation energy": activation_e,
            "enthalpy": enthalpy,
            "breaking bond": breaking,
        }
    )

    df.to_csv(to_path(outname))


if __name__ == "__main__":
    # filename = "~/Documents/Dataset/activation_energy_Green/wb97xd3.csv"
    # plot_prefix = "~/Applications/db_access/greens_rxns"
    # plot_molecules(filename, plot_prefix)

    # filename = "~/Documents/Dataset/activation_energy_Green/wb97xd3_n200.csv"
    filename = "~/Documents/Dataset/activation_energy_Green/wb97xd3.csv"
    outname = "/Users/mjwen/Applications/db_access/greens_rxns/one_bond_break.csv"
    select_one_bond_break_reactions(filename, outname)
