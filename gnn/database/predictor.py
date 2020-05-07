"""
Functions to convert data files to standard files the model accepts.
"""

import logging
import pandas as pd
import multiprocessing
from gnn.database.molwrapper import smiles_to_wrapper_mol, rdkit_mol_to_wrapper_mol
from gnn.database.reaction import Reaction, ReactionCollection
from gnn.utils import expand_path
from gnn.utils import yaml_load, yaml_dump
from rdkit import Chem

logger = logging.getLogger(__name__)


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
            charge_fragment2 should be blank, and the above two format becomes (Don't
            forget the trailing comma):

            reactant,fragment1,,charge_reactant,charge_fragment1,

            or

            reactant,fragment1,

        nprocs (int): number of processors to use to convert smiles to wrapper mol.
            If None, use a serial version.
    """

    def __init__(self, filename, nprocs=None):
        self.filename = expand_path(filename)
        self.nprocs = nprocs
        self.failed = None

    def read_input(self):
        """
        Read reactions specified by smiles given in csv file.

        Returns:
            reactions (list): a sequence of :class:`gnn.database.reaction.Reaction`
        """

        def get_idx(smiles, s, c):
            try:
                idx = smiles[(s, c)]
            except KeyError:
                idx = len(smiles)
                smiles[(s, c)] = idx
            return idx

        df = pd.read_csv(self.filename, header=0, index_col=None)
        num_columns = len(df.columns)
        assert (
            num_columns == 4 or num_columns == 6
        ), f"Corrupted input file; expecting 4 or 6 columns but got {num_columns}"

        # find unique smiles and charge pairs and represent reactions by index in it
        unique_smi_and_cg = {}  # (smiles, charge) as key, index as value
        rxns_by_smi_and_cg_idx = []

        for rxn in df.itertuples():

            # charges not provided
            if num_columns == 4:
                idx, reactant, fragment1, fragment2 = rxn
                charge_r, charge_fg1, charge_fg2 = 0, 0, 0
                if pd.isna(fragment2):
                    fragment2 = None

            # charge provided
            elif num_columns == 6:
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
            else:
                raise RuntimeError("not supported number of columns of input file")

            idx_r = get_idx(unique_smi_and_cg, reactant, int(charge_r))
            idx_p1 = get_idx(unique_smi_and_cg, fragment1, int(charge_fg1))
            if fragment2 is not None:
                idx_p2 = get_idx(unique_smi_and_cg, fragment2, int(charge_fg2))
            else:
                idx_p2 = None
            rxns_by_smi_and_cg_idx.append((idx_r, idx_p1, idx_p2))
        index_to_smi_and_cg = {v: k for k, v in unique_smi_and_cg.items()}

        # convert smiles to wrapper molecules
        smi_and_cg = sorted(unique_smi_and_cg, key=lambda k: unique_smi_and_cg[k])
        if self.nprocs is None:
            molecules = [smiles_to_wrapper_mol(s, c) for s, c in smi_and_cg]
        else:
            with multiprocessing.Pool(self.nprocs) as p:
                molecules = p.starmap(smiles_to_wrapper_mol, smi_and_cg)

        # convert to reactions
        reactions = []
        failed = []  # tuple (bool, failing_reason)
        bad_mol_indices = {i for i, m in enumerate(molecules) if m is None}

        for i, (idx_r, idx_p1, idx_p2) in enumerate(rxns_by_smi_and_cg_idx):
            for idx in (idx_r, idx_p1, idx_p2):
                if idx in bad_mol_indices:
                    failed.append((True, index_to_smi_and_cg[idx]))
                    break
            else:
                failed.append((False, None))

                reactants = [molecules[idx_r]]
                products = [molecules[idx_p1]]
                if idx_p2 is not None:
                    products.append(molecules[idx_p2])
                reactions.append(
                    Reaction(
                        reactants=reactants,
                        products=products,
                        broken_bond=None,
                        free_energy=0.0,  # not used we provide 0.0 taking the place
                        identifier=i,
                    )
                )

        self.failed = failed

        return reactions

    # TODO should directly pass python data struct, instead of files.
    def convert_format(
        self,
        struct_file="struct.sdf",
        label_file="label.yaml",
        feature_file="feature.yaml",
    ):
        """
        Convert to standard files that the fitting code uses.
        """
        reactions = self.read_input()
        extractor = ReactionCollection(reactions)

        extractor.create_struct_label_dataset_reaction_network_based_regression_simple(
            struct_file, label_file, feature_file
        )

    def write_results(self, predictions, filename="bde_result.csv"):
        """
        Append prediction as the last column of a dataframe and write csv file.
        """
        df = pd.read_csv(self.filename, header=0, index_col=None)

        all_predictions = []
        all_failed = []
        p_idx = 0
        for i, (fail, _) in enumerate(self.failed):
            row = df.iloc[i]
            str_row = ",".join([str(i) for i in row])

            # failed at conversion to mol wrapper stage
            if fail:
                all_failed.append(str_row)
                all_predictions.append(None)

            else:
                pred = predictions[p_idx]

                # failed at prediction stage
                if pred is None:
                    all_failed.append(str_row)

                all_predictions.append(pred)
                p_idx += 1

        # if any failed
        if all_failed:
            msg = "\n".join(all_failed)
            print(
                f"\n\nThese reactions failed either at converting smiles to internal "
                f"molecules or converting internal molecules to dgl graph, "
                f"and therefore predictions for them are not made (represented by "
                f"None in the output):\n{msg}\n\n"
            )

        df["bond_energy"] = all_predictions
        filename = expand_path(filename) if filename is not None else filename
        rst = df.to_csv(filename, index=False)
        if rst is not None:
            print(rst)


class PredictionBySDFChargeReactionFiles:
    """
    Make predictions based on the 3 files: molecules.sdf, charges.txt, reactions.csv.

    molecules.sdf: a list of sd data chunks, each representing a molecule.
    charges.txt: charges of the molecules in listed in the molecules file.
    reactions.csv: each line list a reaction (reactant, fragment1, fragment2) by the
    molecule indices (starting from 0) in the molecules.sdf file.
    For example, `0,3,4` means the reactant of the reaction is the first molecule in
    molecules.sdf and its two fragments are the fourth and fifth molecules.
    It is possible that there is only one fragment (e.g. in a ring opening reaction).
    Such reactions can be provided as '0,5,', leaving the 2nd fragment empty.


    Args:
        mol_file (str): molecule file in sdf format, e.g. molecules.sdf
        charge_file (str): charge file, e.g. charges.txt
        rxn_file (str): charge file in csv format, e.g. reactions.csv
        nprocs (int): number of processors to use to convert smiles to wrapper mol.
            If None, use a serial version.
    """

    def __init__(self, mol_file, charge_file, rxn_file, nprocs=None):
        self.mol_file = expand_path(mol_file)
        self.charge_file = expand_path(charge_file)
        self.rxn_file = expand_path(rxn_file)
        self.nprocs = nprocs

    def read_input(self):

        # read sdf mol file
        supp = Chem.SDMolSupplier(self.mol_file, sanitize=True, removeHs=False)
        # a molecule could be None if rdkit cannot process it
        rdkit_mols = [m for m in supp]

        # read charge file
        charges = []
        with open(self.charge_file, "r") as f:
            for line in f:
                charges.append(int(line.strip()))

        # read reaction file
        df_rxns = pd.read_csv(self.rxn_file, header=0, index_col=None)
        num_columns = len(df_rxns.columns)
        assert (
            num_columns == 3
        ), f"Corrupted input file; expecting 3 columns but got {num_columns}"

        # convert rdkit mols to wrapper molecules
        msg = (
            f"expect the number of molecules given in {self.mol_file} and the number "
            f"of charges given in {self.charge_file} to be the same, "
            f"but got {len( rdkit_mols)} and f{len(charges)}. "
        )
        assert len(rdkit_mols) == len(charges), msg

        if self.nprocs is None:
            molecules = [
                rdkit_mol_to_wrapper_mol(m, charge=c, mol_id=i) if m is not None else None
                for i, (m, c) in enumerate(zip(rdkit_mols, charges))
            ]
        else:
            with multiprocessing.Pool(self.nprocs) as p:
                mol_ids = list(range(len(rdkit_mols)))
                args = zip(rdkit_mols, charges, mol_ids)
                molecules = p.starmap(wrapper_rdkit_mol_to_wrapper_mol, args)

        # convert to reactions
        reactions = []
        failed = []  # tuple (bool, failing_reason)
        bad_mol_indices = {i for i, m in enumerate(molecules) if m is None}

        for rxn in df_rxns.itertuples():
            i, idx_r, idx_p1, idx_p2 = rxn
            for idx in (idx_r, idx_p1, idx_p2):
                if idx in bad_mol_indices:
                    failed.append((True, idx))
                    break
            else:
                failed.append((False, None))

                reactants = [molecules[idx_r]]
                products = [molecules[idx_p1]]
                if not pd.isna(idx_p2):
                    products.append(molecules[idx_p2])
                reactions.append(
                    Reaction(
                        reactants=reactants,
                        products=products,
                        broken_bond=None,
                        free_energy=0.0,  # not used we provide 0.0 taking the place
                        identifier=i,
                    )
                )

        self.failed = failed

        return reactions

    # TODO should directly pass python data struct, instead of files.
    def convert_format(
        self,
        struct_file="struct.sdf",
        label_file="label.yaml",
        feature_file="feature.yaml",
    ):
        """
        Convert to standard files that the fitting code uses.
        """
        reactions = self.read_input()
        extractor = ReactionCollection(reactions)

        extractor.create_struct_label_dataset_reaction_network_based_regression_simple(
            struct_file, label_file, feature_file
        )

    def write_results(self, predictions, filename="bde_result.csv"):
        """
        Append prediction as the last column of a dataframe and write csv file.
        """
        df = pd.read_csv(self.rxn_file, header=0, index_col=None)

        all_predictions = []
        all_failed = []
        p_idx = 0
        for i, (fail, reason) in enumerate(self.failed):
            row = df.iloc[i]
            str_row = ",".join([str(i) for i in row])

            # failed at conversion to mol wrapper stage
            if fail:
                logger.info(f"Reaction {i} fails because molecule {reason} fails")
                all_failed.append(str_row)
                all_predictions.append(None)

            else:
                pred = predictions[p_idx]

                # failed at prediction stage
                if pred is None:
                    logger.info(f"Reaction {i} fails because prediction cannot be made")
                    all_failed.append(str_row)

                all_predictions.append(pred)
                p_idx += 1

        # if any failed
        if all_failed:
            msg = "\n".join(all_failed)
            print(
                f"\n\nThese reactions failed either at converting smiles to internal "
                f"molecules or converting internal molecules to dgl graph, "
                f"and therefore predictions for them are not made (represented by "
                f"None in the output):\n{msg}\n\n"
            )

        df["bond_energy"] = all_predictions
        filename = expand_path(filename) if filename is not None else filename
        rst = df.to_csv(filename, index=False)
        if rst is not None:
            print(rst)


class PredictionByStructLabelFeatFiles:
    """
    Make predictions based on the files used by the training script:
    struct.sdf, label.yaml, and feature.yaml.
    Args:
        filename (str): a file containing the path to the three files:
            struct file
            label file
            feature file
    """

    def __init__(self, filename):
        with open(expand_path(filename)) as f:
            lines = f.readlines()
        lines = [expand_path(ln.strip()) for ln in lines]
        self.struct_file = lines[0]
        self.label_file = lines[1]
        self.feature_file = lines[2]

    # TODO should directly pass python data struct, instead of files.
    def convert_format(self):
        return self.struct_file, self.label_file, self.feature_file

    def write_results(self, predictions, filename="bed_result.yaml"):
        """
        Add prediction value as 'prediction' of each reaction (given by a dict) in the
        label file.
        """
        labels = yaml_load(self.label_file)

        failed = []
        for d, p in zip(labels, predictions):
            if p is None:
                failed.append(str(d["id"]))
                d["prediction"] = p
            else:
                d["prediction"] = float(p)

        # if any failed
        if failed:
            msg = ", ".join(failed)
            print(
                f"These reactions failed when converting their molecules, and therefore "
                f"predictions for them are not made: {msg}"
            )

        filename = expand_path(filename) if filename is not None else filename
        if filename is not None:
            yaml_dump(labels, filename)
        else:
            print(labels)


def wrapper_rdkit_mol_to_wrapper_mol(m, *args, **kwargs):
    """
    A rapper around `rdkit_mol_to_wrapper_mol` to deal with m is None case.
    """
    if m is None:
        return None
    else:
        return rdkit_mol_to_wrapper_mol(m * args, **kwargs)
