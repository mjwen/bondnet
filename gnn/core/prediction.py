"""
Converting data files to standard files the model accepts and make predictions.
"""

import os
import logging
import pandas as pd
import json
import multiprocessing
from _collections import OrderedDict
from rdkit import Chem
from pymatgen.analysis.graphs import MoleculeGraph
from gnn.core.molwrapper import (
    MoleculeWrapper,
    rdkit_mol_to_wrapper_mol,
    smiles_to_wrapper_mol,
    inchi_to_wrapper_mol,
)
from gnn.core.reaction import Reaction, create_reactions_from_reactant, factor_integer
from gnn.core.reaction_collection import ReactionCollection
from gnn.utils import expand_path
from gnn.utils import yaml_load, yaml_dump

logger = logging.getLogger(__name__)


class PredictionBase:
    """
    Base class for making predictions.
    """

    def __init__(self):
        pass

    def prepare_data(
        self,
        struct_file="struct.sdf",
        label_file="label.yaml",
        feature_file="feature.yaml",
    ):
        """
        Convert to standard format that the fitting code uses.
        """
        mol = self.read_molecules()
        reactions = self.read_reactions(mol)
        extractor = ReactionCollection(reactions)

        extractor.create_struct_label_dataset_reaction_network_based_regression_simple(
            struct_file, label_file, feature_file
        )

    def write_results(self, predictions, filename=None):
        """
        Write the results given the predictions.

        Args:
            predictions (list): predictions for each reaction return by prepare data.
            filename (str): name for the output file. If None, should output to stdout.
        """
        raise NotImplementedError


class PredictionByOneReactant:
    """
    Make prediction for all bonds in a molecule.

    Args:
        molecule (str): a string representing a molecule.
        format (str): format of the molecule string, supported are `smiles`, `inchi`,
        `sdf`, and `pdb`.
        charge (int): charge of the molecule. If None, inferred from the molecule;
            If provided, it will override the inferred charge.
    """

    def __init__(
        self, molecule, format="smiles", charge=None, allowed_product_charges=None
    ):

        supported_format = {
            "smiles": None,
            "inchi": None,
            "sdf": Chem.MolFromMolBlock,
            "pdb": Chem.MolFromPDBBlock,
        }

        if format not in supported_format:
            supported = ", ".join([k for k in supported_format])
            raise ValueError(
                f"Not supported molecule format `{format}`; choose one of "
                f"{supported} instead."
            )

        self.molecule = molecule
        self.format = format
        self.charge = charge
        self.allowed_product_charges = allowed_product_charges
        self.supported_format = supported_format
        self.failed = None

    def read_molecules(self):
        if self.format == "smiles":
            wrapper_mol = smiles_to_wrapper_mol(self.molecule, self.charge)
        elif self.format == "inchi":
            wrapper_mol = inchi_to_wrapper_mol(self.molecule, self.charge)
        else:
            func = self.supported_format[self.format]
            rdkit_mol = func(self.molecule, sanitize=True, removeHs=False)
            wrapper_mol = rdkit_mol_to_wrapper_mol(rdkit_mol, self.charge)

        self.wrapper_mol = wrapper_mol

        return wrapper_mol

    def read_reactions(self, molecule):
        bonds = [b for b in molecule.bonds]

        reactions = []
        failed = OrderedDict()
        for b in bonds:
            num_products = len(molecule.fragments[b])
            product_charges = factor_integer(
                molecule.charge, self.allowed_product_charges, num_products
            )

            # TODO we choose the first product charges.
            #  this is not a good decision, need to change
            product_charges = [product_charges[0]]

            try:
                # bond energy is not used, we provide 0 taking the place
                rxns, mols = create_reactions_from_reactant(
                    molecule, b, product_charges, bond_energy=0.0
                )
                reactions.extend(rxns)
                failed[b] = (False, None)
            except Chem.AtomKekulizeException:
                reason = "breaking an aromatic bond in ring"
                failed[b] = (True, reason)

        self.failed = failed

        return reactions

    def prepare_data(
        self,
        struct_file="struct.sdf",
        label_file="label.yaml",
        feature_file="feature.yaml",
    ):
        """
        Convert to standard files that the fitting code uses.
        """
        mol = self.read_molecules()
        reactions = self.read_reactions(mol)
        extractor = ReactionCollection(reactions)

        extractor.create_struct_label_dataset_reaction_network_based_regression_simple(
            struct_file, label_file, feature_file
        )

    def write_results(self, predictions, filename=None, to_stdout=True):

        all_predictions = dict()
        all_failed = dict()
        p_idx = 0
        for bond, (fail, reason) in self.failed.items():

            # failed at conversion to wrapper mol stage
            if fail:
                all_failed[bond] = reason
                all_predictions[bond] = None

            else:
                pred = predictions[p_idx]

                # failed at prediction stage
                if pred is None:
                    all_failed[bond] = "cannot convert to dgl graph"

                all_predictions[bond] = pred
                p_idx += 1

        if to_stdout:

            # if any failed
            if all_failed:

                for b, reason in all_failed.items():
                    logger.error(f"Cannot make prediction for bond {b} because {reason}.")

                msg = "\n".join([str(b) for b in all_failed])
                print(
                    f"\n\nFailed breaking bond and creating products, "
                    f"and thus predictions are not made for these bonds:\n{msg}\n"
                    f"See the log file for failing reason.\n\n"
                )

            sdf = add_bond_energy_to_sdf(self.wrapper_mol, all_predictions)
            if filename is None:
                print(
                    f"The bond energies are (last value in lines between `BEGIN BOND` "
                    f"and `End BOND`):\n"
                )
                print(sdf)
            else:
                with open(expand_path(filename), "w") as f:
                    f.write(sdf)
        else:
            return all_predictions


# class PredictionByReactant:
#     """
#     Base class for making predictions using the reactant only, i.e. the products are
#     not given.
#
#     Args:
#         reactants (list): a sequence of MoleculeWrapper molecules
#     """
#
#     def __init__(self, reactants):
#         self.reactants = reactants
#
#     @classmethod
#     def from_smiles(cls, smiles, charges=None):
#         """
#         Args:
#             smiles (list): molecules represented by a list of smiles.
#             charges (list): charges of the molecules. Note smiles have the total charge
#                 built into its notation, i.e. the total charge equals the sum of the
#                 formal charges of atoms or atom groups. If `charges` are provided,
#                 it will override the charges inferred from the smiles.
#         """
#         if charges is not None:
#             assert len(smiles) == len(charges)
#         else:
#             charges = [None] * len(smiles)
#
#         molecules = [smiles_to_wrapper_mol(s, c) for s, c in zip(smiles, charges)]
#
#         return cls(molecules)
#
#     @classmethod
#     def from_sdf(cls, sdf, charges=None):
#         """
#         Args:
#             sdf (list): molecules represented by a list of sdf strings.
#             charges (list): charges of the molecules. If None, inferred from the sdf
#                 strings; If `charges` are provided, it will override the inferred charges.
#         """
#         if charges is not None:
#             assert len(sdf) == len(charges)
#         else:
#             charges = [None] * len(sdf)
#
#         rd_mols = [Chem.MolFromMolBlock(s, sanitize=True, removeHs=False) for s in sdf]
#         molecules = [rdkit_mol_to_wrapper_mol(m, c) for m, c in zip(rd_mols, charges)]
#
#         return cls(molecules)


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
                        identifier=str(i),
                    )
                )

        self.failed = failed

        return reactions

    # TODO should directly pass python data struct, instead of files.
    def prepare_data(
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

    def read_molecules(self):

        # read sdf mol file
        supp = Chem.SDMolSupplier(self.mol_file, sanitize=True, removeHs=False)
        # a molecule could be None if rdkit cannot process it
        rdkit_mols = [m for m in supp]

        # read charge file
        charges = []
        with open(self.charge_file, "r") as f:
            for line in f:
                charges.append(int(line.strip()))

        # convert rdkit mols to wrapper molecules
        msg = (
            f"expect the number of molecules given in {self.mol_file} and the number "
            f"of charges given in {self.charge_file} to be the same, "
            f"but got {len( rdkit_mols)} and f{len(charges)}. "
        )
        assert len(rdkit_mols) == len(charges), msg

        if self.nprocs is None:
            molecules = [
                rdkit_mol_to_wrapper_mol(m, charge=c, identifier=str(i))
                if m is not None
                else None
                for i, (m, c) in enumerate(zip(rdkit_mols, charges))
            ]
        else:
            with multiprocessing.Pool(self.nprocs) as p:
                mol_ids = list(range(len(rdkit_mols)))
                args = zip(rdkit_mols, charges, mol_ids)
                molecules = p.starmap(wrapper_rdkit_mol_to_wrapper_mol, args)

        return molecules

    def read_reactions(self, molecules):

        # read reaction file
        df_rxns = pd.read_csv(self.rxn_file, header=0, index_col=None)
        num_columns = len(df_rxns.columns)
        assert (
            num_columns == 3
        ), f"Corrupted input file; expecting 3 columns but got {num_columns}"

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
                        identifier=str(i),
                    )
                )

        self.failed = failed

        return reactions

    # TODO should directly pass python data struct, instead of files.
    def prepare_data(
        self,
        struct_file="struct.sdf",
        label_file="label.yaml",
        feature_file="feature.yaml",
    ):
        """
        Convert to standard files that the fitting code uses.
        """
        molecules = self.read_molecules()
        reactions = self.read_reactions(molecules)
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


class PredictionByMolGraphReactionFiles(PredictionBySDFChargeReactionFiles):
    """
    Make predictions based on the two files: molecules.json (or molecules.yaml) and 
        reactions.csv.

        molecules.json (or molecules.yaml) stores all the molecules in the reactions
        and it should be a list of MoleculeGraph.as_dict().
    """

    def __init__(self, mol_file, rxn_file, nprocs=None):
        self.mol_file = expand_path(mol_file)
        self.rxn_file = expand_path(rxn_file)
        self.nprocs = nprocs

    def read_molecules(self):

        file_type = os.path.splitext(self.mol_file)[1]
        if file_type == ".json":
            with open(self.mol_file, "r") as f:
                mol_graph_dicts = json.load(f)
        elif file_type in [".yaml", ".yml"]:
            mol_graph_dicts = yaml_load(self.mol_file)
        else:
            supported = [".json", ".yaml", ".yml"]
            raise ValueError(
                f"File extension of {self.mol_file} not supported; "
                f"supported are: {supported}."
            )

        mol_graphs = [MoleculeGraph.from_dict(d) for d in mol_graph_dicts]
        molecules = [
            MoleculeWrapper(g, free_energy=0.0, id=str(i))
            for i, g in enumerate(mol_graphs)
        ]

        return molecules


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
    def prepare_data(self):
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


def add_bond_energy_to_sdf(m, energy):
    """
    Add the bond energies of a molecule to sdf v3000 file.

    Args:
        m (MoleculeWrapper): the molecule
        energy (dict): bond energies for molecule, with bond index (a 2-tuple) as key
            and bond energy as value.

    Returns:
        str: an sdf v3000 file with bond energies
    """
    sdf = m.write(v3000=True)
    bonds = m.get_sdf_bond_indices(zero_based=True, sdf=sdf)

    lines = sdf.split("\n")
    start = end = 0
    for i, ln in enumerate(lines):
        if "BEGIN BOND" in ln:
            start = i + 1
        if "END BOND" in ln:
            end = i
            break

    for ib, i in enumerate(range(start, end)):
        e = energy[bonds[ib]]
        if e is None:
            e = ""
        lines[i] += f"  {e}"

    sdf = "\n".join(lines)

    return sdf
