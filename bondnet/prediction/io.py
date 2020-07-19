"""
Converting data files to standard data the model accepts and write prediction results.
"""

import os
import logging
import json
import multiprocessing
import warnings
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from pymatgen.analysis.graphs import MoleculeGraph
from bondnet.core.molwrapper import MoleculeWrapper, rdkit_mol_to_wrapper_mol
from bondnet.core.rdmol import (
    smiles_to_rdkit_mol,
    inchi_to_rdkit_mol,
    RdkitMolCreationError,
    read_rdkit_mols_from_file,
)
from bondnet.core.reaction import Reaction, ReactionExtractorFromReactant
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import expand_path
from bondnet.utils import yaml_load, yaml_dump

logger = logging.getLogger(__name__)


class BasePrediction:
    """
    Base class for making predictions.
    """

    def __init__(self):
        self._molecules = None
        self._reactions = None
        self._no_result_reason = None

    @property
    def molecules(self):
        """
        Returns:
            list: MoleculeWrapper molecules.
        """
        return self._molecules

    @property
    def reactions(self):
        """
        Returns:
            list: a sequence of Reaction
        """
        return self._reactions

    @property
    def no_result_reason(self):
        """
        Returns:
            dict: {reaction_index:reason} the reason why some request reactions do not
                have result
        """
        return self._no_result_reason

    def read_molecules(self):
        """
        Read molecules from the input, a file or string.

        Returns:
            list: MoleculeWrapper molecules.
        """

        raise NotImplementedError

    def read_reactions(self):
        """
         Read reactions from the input, a file, string, or creating it by breaking
         bonds in reactant.

        Returns:
            list: a sequence of Reaction
        """
        raise NotImplementedError

    def prepare_data(self):
        """
        Convert to standard format that the fitting code uses.
        """
        self.read_molecules()
        self.read_reactions()
        extractor = ReactionCollection(self.reactions)

        mols, labls, feats = extractor.create_regression_dataset_reaction_network_simple(
            write_to_file=False
        )

        return mols, labls, feats

    def write_results(self, predictions, filename):
        """
        Write the results given the predictions.

        Args:
            predictions (list): predictions for each reaction return by prepare data.
            filename (str): name for the output file. If None, write to stdout.
        """
        raise NotImplementedError


class PredictionOneReactant(BasePrediction):
    """
    Make prediction for all bonds in a molecule.

    Args:
        molecule (str): a string representing a molecule.
        format (str): format of the molecule string, supported are `smiles`, `inchi`,
        `sdf`, and `pdb`.
        charge (int): charge of the molecule. 
        allowed_product_charges (list): allowed charges for created product molecules
        ring_bond (bool): whether to make predictions for ring bond
    """

    def __init__(
        self,
        molecule,
        charge=0,
        format="smiles",
        allowed_product_charges=[0],
        ring_bond=False,
    ):
        super(PredictionOneReactant, self).__init__()

        self.molecule_str = molecule
        self.format = format
        self.charge = charge
        self.allowed_product_charges = allowed_product_charges
        self.ring_bond = ring_bond

        self.rxn_idx_to_bond_map = None

    def read_molecules(self):
        if self.format == "smiles":
            rdkit_mol = smiles_to_rdkit_mol(self.molecule_str)
        elif self.format == "inchi":
            rdkit_mol = inchi_to_rdkit_mol(self.molecule_str)
        elif self.format == "sdf":
            rdkit_mol = Chem.MolFromMolBlock(
                self.molecule_str, sanitize=True, removeHs=False
            )
            if rdkit_mol is None:
                raise RdkitMolCreationError(f"{self.format}")
        elif self.format == "pdb":
            rdkit_mol = Chem.MolFromPDBBlock(
                self.molecule_str, sanitize=True, removeHs=False
            )
            if rdkit_mol is None:
                raise RdkitMolCreationError(f"{self.format}")
        else:
            raise ValueError(
                f"Not supported molecule format `{format}`; choose one of "
                f"`smiles`, `inchi`, `sdf` or `pdb` instead."
            )

        wrapper_mol = rdkit_mol_to_wrapper_mol(
            rdkit_mol, self.charge, identifier=rdkit_mol.GetProp("_Name")
        )

        self._molecules = [wrapper_mol]

        return self._molecules

    def read_reactions(self):
        molecule = self.molecules[0]

        extractor = ReactionExtractorFromReactant(
            molecule, allowed_charge=self.allowed_product_charges
        )
        extractor.extract(ring_bond=self.ring_bond, one_per_iso_bond_group=True)

        reactions = extractor.reactions
        for r in reactions:
            r.set_free_energy(0.0)

        self._reactions = reactions
        self._no_result_reason = extractor.no_reaction_reason
        self.rxn_idx_to_bond_map = extractor.rxn_idx_to_bond_map

        return reactions

    def write_results(self, predictions, figure_name, to_stdout=False):

        # group prediction by bond
        predictions_by_bond = defaultdict(list)
        for i, p in enumerate(predictions):
            bond = self.rxn_idx_to_bond_map[i]
            predictions_by_bond[bond].append(p)

        # obtain smallest energy for each bond across charge
        for bond, pred in predictions_by_bond.items():
            pred = [p for p in pred if p is not None]
            # all prediction of the same bond across charges are None
            if not pred:
                predictions_by_bond[bond] = None
            # at least one value is not None
            else:
                predictions_by_bond[bond] = min(pred)

        # create prediction and failing info for all bonds
        all_predictions = dict()
        all_failed = dict()
        for bond, (compute, fail, reason) in self.no_result_reason.items():

            if not compute:
                all_predictions[bond] = None

            # failed at conversion to wrapper mol stage
            elif fail:
                all_failed[bond] = reason
                all_predictions[bond] = None

            else:
                pred = predictions_by_bond[bond]

                # failed at prediction stage
                if pred is None:
                    all_failed[bond] = "cannot convert to dgl graph"

                all_predictions[bond] = pred

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

        sdf = add_bond_energy_to_sdf(self.molecules[0], all_predictions)
        bond_note = {k: v for k, v in all_predictions.items() if v is not None}
        self.molecules[0].draw_with_bond_note(bond_note, filename=figure_name)

        if to_stdout:
            print(sdf)

            print(
                "The predicted bond energies in the SDF file are the 7th value in lines "
                "between `BEGIN BOND` and `End BOND`."
            )
            print(f"Also shown in the generated file `{figure_name}`.")

        return sdf


class PredictionMultiReactant(BasePrediction):
    """
    Make prediction for all bonds in a multiple molecules.

    Args:
        molecule_file (str): file listing all the molecules
        charge_file (str): charges of molecule listed in molecules. If None,
            molecule charge is inferred from the molecule. If provided,
            it will override the inferred charge.
        format (str): format of the molecule in the molecule_file, supported are
            `smiles`, `inchi`, `sdf`, and `pdb`.
        allowed_product_charges (list): allowed charges for created product molecules
        ring_bond (bool): whether to make predictions for ring bond
    """

    def __init__(
        self,
        molecule_file,
        charge_file=None,
        format="smiles",
        allowed_product_charges=[0],
        ring_bond=False,
    ):
        super(PredictionMultiReactant, self).__init__()

        self.molecule_file = expand_path(molecule_file)
        self.charge_file = expand_path(charge_file) if charge_file is not None else None
        self.format = format
        self.allowed_product_charges = allowed_product_charges
        self.ring_bond = ring_bond

        self.rxn_idx_to_mol_and_bond_map = None

    def read_molecules(self):

        # read rdkit mols
        rdkit_mols = read_rdkit_mols_from_file(self.molecule_file, self.format)

        # read charge file
        if self.charge_file is None:
            charges = [0] * len(rdkit_mols)
        else:
            charges = read_charge(self.charge_file)

            msg = (
                f"expect the number of molecules given in {self.molecule_file} and the "
                f"number of charges given in {self.charge_file} to be the same, "
                f"but got {len(rdkit_mols)} and f{len(charges)}. "
            )
            assert len(rdkit_mols) == len(charges), msg

        # convert rdkit mols to wrapper molecules
        identifiers = [
            m.GetProp("_Name") + f"_index-{i}" if m is not None else None
            for i, m in enumerate(rdkit_mols)
        ]
        self._molecules = rdkit_mols_to_wrapper_mols(rdkit_mols, identifiers, charges)

        return self._molecules

    def read_reactions(self):

        reactions = []
        no_result_reason = []
        rxn_idx_to_mol_and_bond_map = {}

        rxn_idx = 0
        for im, mol in enumerate(self.molecules):

            if mol is None:
                no_result_reason.append(None)
            else:
                extractor = ReactionExtractorFromReactant(
                    mol, allowed_charge=self.allowed_product_charges
                )
                extractor.extract(ring_bond=False, one_per_iso_bond_group=True)

                rxns = extractor.reactions
                reactions.extend(rxns)

                for i in range(len(rxns)):
                    bond = extractor.rxn_idx_to_bond_map[i]
                    rxn_idx_to_mol_and_bond_map[rxn_idx] = (im, bond)
                    rxn_idx += 1

                no_result_reason.append(extractor.no_reaction_reason)

        for r in reactions:
            r.set_free_energy(0.0)

        self._reactions = reactions
        self._no_result_reason = no_result_reason
        self.rxn_idx_to_mol_and_bond_map = rxn_idx_to_mol_and_bond_map

        return reactions

    def write_results(self, predictions, filename=None):

        # group predictions by molecule and bond
        predictions_by_mol_and_bond = defaultdict(lambda: defaultdict(list))
        for i, p in enumerate(predictions):
            mol_idx, bond = self.rxn_idx_to_mol_and_bond_map[i]
            predictions_by_mol_and_bond[mol_idx][bond].append(p)

        # obtain smallest energy for each bond across charges
        for mol_id in predictions_by_mol_and_bond:
            for bond, pred in predictions_by_mol_and_bond[mol_id].items():
                pred = [p for p in pred if p is not None]

                # all predictions of the same bond across charges are None
                if not pred:
                    predictions_by_mol_and_bond[mol_id][bond] = None

                # at least one value is not None
                else:
                    predictions_by_mol_and_bond[mol_id][bond] = min(pred)

        # create prediction and failing info
        all_predictions = []
        all_failed = []

        for mol_id, x in enumerate(self.no_result_reason):
            if x is None:

                # failed at converting to mol stage
                all_predictions.append(None)
                all_failed.append(None)

            else:
                predictions = dict()
                failed = dict()

                for bond, (compute, fail, reason) in x.items():

                    if not compute:
                        predictions[bond] = None

                    # failed at conversion to wrapper mol stage
                    elif fail:
                        failed[bond] = reason
                        predictions[bond] = None

                    else:
                        pred = predictions_by_mol_and_bond[mol_id][bond]

                        # failed at prediction stage
                        if pred is None:
                            failed[bond] = "cannot convert to dgl graph"

                        predictions[bond] = pred

                all_predictions.append(predictions)
                all_failed.append(failed)

        has_failed = True
        for failed in all_failed:
            if failed is None:
                logger.error(f"Cannot read molecule {mol_id}, ignored.")
                has_failed = True
            else:
                if failed:
                    has_failed = True
                for b, reason in failed.items():
                    logger.error(
                        f"Cannot make prediction for bond {b} of molecule {mol_id} "
                        f"because {reason}."
                    )

        if has_failed:
            print(
                f"\n\nPrediction cannot be made for some molecules. See the log file "
                f"for failing reason.\n\n"
            )

        # write results to sdf file
        all_sdf = []
        for i, predictions in enumerate(all_predictions):
            if predictions is not None:
                sdf = add_bond_energy_to_sdf(self.molecules[i], predictions)
                all_sdf.append(sdf)
        all_sdf = "".join(all_sdf)

        if filename is None:
            print(all_sdf)
        else:
            with open(expand_path(filename), "w") as f:
                f.write(all_sdf)
            print(f"The predictions have been written to file {filename}.\n")
        print(
            f"The predicted bond energies are the 7th value in lines between "
            f"`BEGIN BOND` and `End BOND`.\n"
        )

        return all_predictions


class PredictionByReaction(BasePrediction):
    """
    Make predictions for bonds given as reactions.

    Three files are needed: molecules.extension, reactions.csv, and charges.txt.
    The first two are mandatory, giving all the molecules and the reactions they
    can form, respectively. The third charges.txt file is optional if all molecules
    have charge 0 or the charge info can be obtained from the molecules (e.g. when
    pymatgen molecule graph is used).

    Args:
        mol_file (str): path to molecule file, e.g. mols.sdf, mols.pdb
        rxn_file (str): path to charge file in csv format, e.g. reactions.csv
        charge_file (str): charge file, e.g. charges.txt
        format (str): format of the molecule file (e.g. `sdf`, `graph`, `pdb`,
            `smiles`, `inchi`.
        nprocs (int): number of processors to use to convert smiles to wrapper mol.
            If None, use a serial version.
    """

    def __init__(
        self, molecule_file, reaction_file, charge_file=None, format="sdf", nprocs=None
    ):
        self.molecule_file = expand_path(molecule_file)
        self.reaction_file = expand_path(reaction_file)
        self.charge_file = expand_path(charge_file) if charge_file is not None else None
        self.format = format
        self.nprocs = nprocs

    def read_molecules(self):

        if self.format == "graph":

            if self.charge_file is not None:
                warnings.warn(
                    f"charge file {self.charge_file} ignored for format `graph`"
                )

            file_type = os.path.splitext(self.molecule_file)[1]
            if file_type == ".json":
                with open(self.molecule_file, "r") as f:
                    mol_graph_dicts = json.load(f)
            elif file_type in [".yaml", ".yml"]:
                mol_graph_dicts = yaml_load(self.molecule_file)
            else:
                supported = [".json", ".yaml", ".yml"]
                raise ValueError(
                    f"File extension of {self.molecule_file} not supported; "
                    f"supported are: {supported}."
                )

            mol_graphs = [MoleculeGraph.from_dict(d) for d in mol_graph_dicts]
            molecules = [MoleculeWrapper(g, id=str(i)) for i, g in enumerate(mol_graphs)]

        else:
            # read rdkit mols
            rdkit_mols = read_rdkit_mols_from_file(self.molecule_file, self.format)

            # read charge file
            if self.charge_file is None:
                charges = [0] * len(rdkit_mols)
            else:
                charges = read_charge(self.charge_file)
                msg = (
                    f"expect the number of molecules given in {self.molecule_file} "
                    f"and the number of charges given in {self.charge_file} to be "
                    f"the same, but got {len(rdkit_mols)} and f{len(charges)}. "
                )
                assert len(rdkit_mols) == len(charges), msg

            # convert rdkit mols to wrapper molecules
            identifiers = [
                m.GetProp("_Name") + f"_index-{i}" if m is not None else None
                for i, m in enumerate(rdkit_mols)
            ]
            molecules = rdkit_mols_to_wrapper_mols(
                rdkit_mols, identifiers, charges, nprocs=self.nprocs
            )

        self._molecules = molecules

        return molecules

    def read_reactions(self):

        # read reaction file
        df_rxns = pd.read_csv(self.reaction_file, header=0, index_col=None)
        num_columns = len(df_rxns.columns)
        assert (
            num_columns == 3
        ), f"Corrupted input file; expecting 3 columns but got {num_columns}"

        # convert to reactions
        bad_mol_indices = {i for i, m in enumerate(self.molecules) if m is None}

        reactions = []
        no_result_reason = []  # each element is a tuple (fail, failing_reason)

        for rxn in df_rxns.itertuples():
            i, idx_r, idx_p1, idx_p2 = rxn
            for idx in (idx_r, idx_p1, idx_p2):
                if idx in bad_mol_indices:
                    no_result_reason.append((True, idx))
                    break
            else:
                reactants = [self.molecules[idx_r]]
                products = [self.molecules[idx_p1]]

                # two products
                if not pd.isna(idx_p2):
                    products.append(self.molecules[idx_p2])

                reactions.append(
                    Reaction(
                        reactants=reactants,
                        products=products,
                        broken_bond=None,
                        free_energy=0.0,  # not used; provide 0 taking the place
                        identifier=str(i),
                    )
                )

                no_result_reason.append((False, None))

        self._reactions = reactions
        self._no_result_reason = no_result_reason

        return reactions

    def write_results(self, predictions, filename="result.csv"):
        """
        Append prediction as the last column of a dataframe and write csv file.
        """

        df = pd.read_csv(self.reaction_file, header=0, index_col=None)

        all_predictions = []
        all_failed = []
        p_idx = 0
        for i, (fail, reason) in enumerate(self.no_result_reason):
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
                f"None in the output):\n{msg}\n\n."
                f"See the log file for more info."
            )

        df["bond_energy"] = all_predictions
        filename = expand_path(filename) if filename is not None else filename
        rst = df.to_csv(filename, index=False)
        if rst is not None:
            print(rst)


class PredictionSmilesReaction:
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
            reactions (list): a sequence of :class:`bondnet.database.reaction.Reaction`
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
        rdkit_mols = []
        for s, c in smi_and_cg:
            try:
                m = smiles_to_rdkit_mol(s)
            except RdkitMolCreationError:
                m = None
            rdkit_mols.append((m, c, None, s))
        if self.nprocs is None:
            molecules = [wrapper_rdkit_mol_to_wrapper_mol(*x) for x in rdkit_mols]
        else:
            with multiprocessing.Pool(self.nprocs) as p:
                molecules = p.starmap(wrapper_rdkit_mol_to_wrapper_mol, rdkit_mols)

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

    def prepare_data(self):
        reactions = self.read_input()
        extractor = ReactionCollection(reactions)

        out = extractor.create_regression_dataset_reaction_network_simple(
            write_to_file=False
        )

        return out

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


class PredictionStructLabelFeatFiles:
    """
    Make predictions based on the files used by the training script:
    struct.sdf, label.yaml, and feature.yaml.

    Args:
        struct_file (str):
        label_file (str):
        feature_file (str):
    """

    def __init__(self, struct_file, label_file, feature_file):
        self.struct_file = expand_path(struct_file)
        self.label_file = expand_path(label_file)
        self.feature_file = expand_path(feature_file)

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


def rdkit_mols_to_wrapper_mols(
    rdkit_mols, identifiers, charges=None, energies=None, nprocs=None
):
    """
    Convert a list of rdkit molecules to MoleculeWrapper molecules.

    Args:
        rdkit mols (list): rdkit molecule

    Returns:
        list: MoleculeWrapper molecule
    """
    charges = [None] * len(rdkit_mols) if charges is None else charges
    energies = [None] * len(rdkit_mols) if energies is None else energies

    if nprocs is None:
        molecules = [
            wrapper_rdkit_mol_to_wrapper_mol(m, c, e, iden)
            for m, c, e, iden in zip(rdkit_mols, charges, energies, identifiers)
        ]
    else:
        with multiprocessing.Pool(nprocs) as p:
            args = zip(rdkit_mols, charges, energies, identifiers)
            molecules = p.starmap(wrapper_rdkit_mol_to_wrapper_mol, args)

    return molecules


def wrapper_rdkit_mol_to_wrapper_mol(m, *args, **kwargs):
    """
    A rapper around `rdkit_mol_to_wrapper_mol` to deal with m is None case.
    """
    if m is None:
        return None
    else:
        return rdkit_mol_to_wrapper_mol(m, *args, **kwargs)


def read_charge(filename):
    """
    Read charges of molecule from file, one charge per line.

    Returns:
        list: charges of molecules
    """
    charges = []
    with open(filename, "r") as f:
        for line in f:
            charges.append(int(line.strip()))

    return charges
