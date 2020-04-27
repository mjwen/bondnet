"""
Functions to convert data files to standard files the model accepts.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from gnn.database.molwrapper import rdkit_mol_to_wrapper_mol
from gnn.database.reaction import Reaction, ReactionCollection
from gnn.utils import expand_path


class PredictionBySmilesReaction:
    """
    Read reactions in which reactants and products are given in smiles in a csv file.

    Args:
    filename (str): csv file containing the reaction file, with headers:

            reactant,fragment1,fragment2,charge_reactant,charge_fragment1,charge_fragment2

            or

            reactant,fragment1,fragment2

            The latter assumes charge of all molecules are zero.

            If there is only one fragment (e.g. break a bond in a ring), fragment2 and
            charge_fragments should be blank, and the above two format becomes (Don't
            forget the trailing comma):

            reactant,fragment1,,charge_reactant,charge_fragment1,

            or

            reactant,fragment1,
    """

    def __init__(self, filename):
        self.filename = expand_path(filename)

    def read_smiles_csv(self):
        """
        Read reactions specified by smiles given in csv file.

        Returns:
            reactions (list): a sequence of :class:`gnn.database.reaction.Reaction`
        """

        df = pd.read_csv(self.filename, header=0, index_col=None)

        def get_mol(smiles_and_charge, molecules, s, cg):
            """
            get molecule from molecule reservior; create if not exist
            """
            try:
                idx = smiles_and_charge.index((s, cg))
                m = molecules[idx]

            except ValueError:  # not in mol reservoir

                # try:
                # create molecules
                m = Chem.AddHs(Chem.MolFromSmiles(s))
                AllChem.EmbedMolecule(m, randomSeed=35)
                m = rdkit_mol_to_wrapper_mol(m, charge=cg, mol_id=len(molecules))

                # m = pybel.readstring("smi", s).OBMol
                # m.AddHydrogens()
                # m = ob_mol_to_wrapper_mol(m, charge=0, mol_id=len(molecules))

                molecules.append(m)
                smiles_and_charge.append((s, cg))

            # except ValueError:  # cannot convert smiles string to mol
            #     m = None

            return m

        # convert to reactions
        # molecules with same smiles string will be created once and shared
        molecule_smiles_and_charge = []
        molecules = []
        reactions = []

        num_columns = len(df.columns)
        for rxn in df.itertuples():

            if num_columns == 4:  # charges not provided
                idx, reactant, fragment1, fragment2 = rxn
                charge_r, charge_fg1, charge_fg2 = 0, 0, 0
                if pd.isna(fragment2):
                    fragment2 = None
            else:  # charge provided
                (
                    idx,
                    reactant,
                    fragment1,
                    fragment2,
                    charge_r,
                    charge_fg1,
                    charge_fg2,
                ) = rxn
                if pd.isna(fragment2) or pd.isna(charge_fg2):
                    fragment2 = None

            r = get_mol(molecule_smiles_and_charge, molecules, reactant, int(charge_r))
            p1 = get_mol(
                molecule_smiles_and_charge, molecules, fragment1, int(charge_fg1)
            )
            if fragment2 is not None:
                p2 = get_mol(
                    molecule_smiles_and_charge, molecules, fragment2, int(charge_fg2)
                )

            reactants = [r]
            products = [p1, p2] if fragment2 is not None else [p1]
            reactions.append(
                # free_energy is not used, we provide 0.0 taking the place
                Reaction(
                    reactants,
                    products,
                    broken_bond=None,
                    free_energy=0.0,
                    identifier=idx,
                )
            )

        return reactions

    # TODO should directly pass python data struct, instead of files.
    def convert_smiles_csv(
        self,
        struct_file="struct.sdf",
        label_file="label.yaml",
        feature_file="feature.yaml",
    ):
        """
        Convert to standard files that the fitting code uses.
        """
        reactions = self.read_smiles_csv()
        extractor = ReactionCollection(reactions)

        extractor.create_struct_label_dataset_reaction_network_based_regression_simple(
            struct_file, label_file, feature_file
        )

    def write_results(self, predictions, filename):
        """
        Append prediction as the last column of a dataframe and write csv file.
        """
        df = pd.read_csv(self.filename, header=0, index_col=None)
        df["bond_energy"] = predictions
        filename = expand_path(filename) if filename is not None else filename
        rst = df.to_csv(filename, index=False)
        if rst is not None:
            print(rst)
