import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from gnn.database.molwrapper import rdkit_mol_to_wrapper_mol
from gnn.database.reaction import Reaction
from gnn.utils import expand_path


logger = logging.getLogger(__name__)


def read_nrel_bde_dataset(filename):
    """
    Read the zinc bde dataset.

    The dataset is described in:
    https://chemrxiv.org/articles/Prediction_of_Homolytic_Bond_Dissociation_Enthalpies_for_Organic_Molecules_at_near_Chemical_Accuracy_with_Sub-Second_Computational_Cost/10052048/2

    Args:
        filename (str): csv file containing the bde info

    Returns:
        mols (list): a sequence of :class:`MoleculeWrapper`.
        energies (list of dict): bond energies. Each dict for one molecule, with bond
            index (a tuple) as key and bond energy as value.
    """

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

    print("Number of reactions:", df.shape[0])
    print("Duplicate reactions:", df.shape[0] - len(selected_rxns))
    print("Remaining reactions:", len(selected_rxns))

    def get_mol(smiles, molecules, s):
        """get molecules; create if not exist
        """
        try:
            idx = smiles.index(s)
            m = molecules[idx]

        except ValueError:  # not in mol reservoir

            # try:
            # create molecules
            m = Chem.AddHs(Chem.MolFromSmiles(s))
            AllChem.EmbedMolecule(m, randomSeed=35)
            m = rdkit_mol_to_wrapper_mol(m, charge=0, mol_id=len(molecules))

            # m = pybel.readstring("smi", s).OBMol
            # m.AddHydrogens()
            # m = ob_mol_to_wrapper_mol(m, charge=0, mol_id=len(molecules))

            molecules.append(m)
            smiles.append(s)

        # except ValueError:  # cannot convert smiles string to mol
        #     m = None

        return m

    # convert to reactions
    # molecules with same smiles string will be created once and shared
    molecule_smiles = []
    molecules = []
    reactions = []
    for rxn in selected_rxns:
        idx, rid, reactant, bond_index, product1, product2, bde, bond_type = rxn
        r = get_mol(molecule_smiles, molecules, reactant)
        if r is None:
            continue
        p1 = get_mol(molecule_smiles, molecules, product1)
        if p1 is None:
            continue
        p2 = get_mol(molecule_smiles, molecules, product2)
        if p2 is None:
            continue

        reactions.append(
            Reaction(
                reactants=[r],
                products=[p1, p2],
                broken_bond=None,
                free_energy=bde,
                identifier=rid,
            )
        )

    print("Number of eactions after conversion:", len(reactions))
    print("Unique molecules in all reactions:", len(molecules))

    return reactions


if __name__ == "__main__":

    reactions = read_nrel_bde_dataset(
        "~/Documents/Dataset/NREL_BDE/rdf_data_190531_n200.csv"
    )
