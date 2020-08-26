import os
import logging
import argparse
from bondnet.core.rdmol import read_rdkit_mols_from_file
from bondnet.core.molwrapper import rdkit_mol_to_wrapper_mol
from bondnet.core.reaction import Reaction
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import yaml_load

logger = logging.getLogger(__name__)


def create_label_file(
    molecule_file="molecule.sdf",
    molecule_attributes_file="molecule_attributes.yaml",
    reaction_file="reaction.yaml",
    label_file="label.yaml",
):
    """
    Convert input molecule, attributes, and reaction file to an atom mapped label file.

    The molecule, attributes, and label file can then be used by the training code.

    See the `examples/train` directory for example molecule, attributes, and reaction file.

    Args:
        molecule_file (str): path to the input molecule file
        molecule_attributes_file (str): path to the input molecule attributes file
        reaction_file (str): path to the reaction file
        label_file (str or None): path to the output label file. If `None` no file will
            be written, but the molecules and reactions will be directly returned.
    """
    molecules = _read_molecules(molecule_file, molecule_attributes_file)
    reactions = _read_reactions(molecules, reaction_file)
    rxn_coll = ReactionCollection(reactions)

    if label_file is not None:
        out = rxn_coll.create_regression_dataset_reaction_network_simple(
            struct_file=os.devnull, label_file=label_file, feature_file=None
        )
    else:

        out = rxn_coll.create_regression_dataset_reaction_network_simple(
            write_to_file=False
        )

    rdmols, rxns, attrs = out

    return rdmols, attrs, rxns


def read_input_files(molecule_file, molecule_attributes_file, reaction_file):
    """
    A wrapper function to determine whether to convert the reaction file or not.

    This function behaves differently depending on the reaction_file.
    If it is a raw csv file without atom mapping, `create_label_file()` is called to
    convert the file. If it is already an atom-mapped file, simply return it.

    Args:
        molecule_file (str): path to the input molecule file
        molecule_attributes_file (str): path to the input molecule attributes file
        reaction_file (str): path to the reaction file.

    Returns:
        mols
        attrs
        rxns
    """

    atom_mapped = False
    with open(reaction_file) as f:
        for line in f:
            if "atom_mapping" in line:
                atom_mapped = True
                break

    if atom_mapped:
        return molecule_file, molecule_attributes_file, reaction_file
    else:
        mols, attrs, rxns = create_label_file(
            molecule_file, molecule_attributes_file, reaction_file, label_file=None
        )

        return mols, attrs, rxns


def _read_molecules(molecule_file, molecule_attributes_file):

    # read rdkit mols
    rdkit_mols = read_rdkit_mols_from_file(molecule_file)

    # read molecule attributes
    attrs = yaml_load(molecule_attributes_file)
    msg = (
        f"expect the number of molecules given in {molecule_file} "
        f"and the number of molecule attributes given in {molecule_attributes_file} to "
        f"be the same, but got {len(rdkit_mols)} and f{len(attrs)}. "
    )
    assert len(rdkit_mols) == len(attrs), msg

    # convert rdkit mols to wrapper molecules
    identifiers = [
        m.GetProp("_Name") + f"_index-{i}" if m is not None else None
        for i, m in enumerate(rdkit_mols)
    ]
    charges = [a["charge"] for a in attrs]
    molecules = [
        rdkit_mol_to_wrapper_mol(m, charge=cg, free_energy=None, identifier=idx)
        for m, idx, cg in zip(rdkit_mols, identifiers, charges)
    ]

    return molecules


def _read_reactions(molecules, reaction_file):

    # read reaction file
    rxns = yaml_load(reaction_file)

    # convert to reactions
    bad_mol_indices = {i for i, m in enumerate(molecules) if m is None}

    reactions = []
    no_result_reason = []  # each element is a tuple (fail, failing_reason)

    for i, rxn in enumerate(rxns):
        idx_r = rxn["reactants"][0]
        idx_p1 = rxn["products"][0]
        if len(rxn["products"]) == 1:
            idx_p2 = None
        else:
            idx_p2 = rxn["products"][1]
        bde = rxn["energy"]

        for idx in (idx_r, idx_p1, idx_p2):
            if idx in bad_mol_indices:
                no_result_reason.append((True, idx))
                break
        else:
            reactants = [molecules[idx_r]]
            products = [molecules[idx_p1]]
            if idx_p2 is not None:
                products.append(molecules[idx_p2])

            reactions.append(
                Reaction(
                    reactants=reactants,
                    products=products,
                    broken_bond=None,
                    free_energy=bde,
                    identifier=f"reaction_{i}",
                )
            )

            no_result_reason.append((False, None))

    for i, reason in enumerate(no_result_reason):
        if reason[0]:
            logger.warning(
                f"Reaction {i} ignored because failed to read molecule {reason[1]}."
            )

    return reactions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate label file from a molecule file, a molecule attributes "
        "file, and a reaction file. This will also creates the atom mapping between "
        "the reactant and the products. See the `examples` directory for examples of "
        "the three files."
    )
    parser.add_argument("molecule_file", type=str, help="a sdf file of molecules")
    parser.add_argument(
        "molecule_attributes_file", type=str, help="a yaml file of molecule attributes"
    )
    parser.add_argument("reaction_file", type=str, help="a csv file of reactions")
    parser.add_argument(
        "--label-file",
        type=str,
        default="label.yaml",
        help="the output yaml file of labels",
    )

    args = parser.parse_args()

    create_label_file(
        args.molecule_file,
        args.molecule_attributes_file,
        args.reaction_file,
        args.label_file,
    )
