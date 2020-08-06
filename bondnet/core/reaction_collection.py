import itertools
import logging
import multiprocessing
import numpy as np
from collections import defaultdict, OrderedDict
from bondnet.core.reaction import Reaction, ReactionsMultiplePerBond, ReactionsOnePerBond
from bondnet.utils import create_directory, pickle_load, yaml_dump, to_path

logger = logging.getLogger(__name__)


class ReactionCollection:
    """
    A set of Reactions, and operations on them.
    """

    def __init__(self, reactions):
        """
        Args:
            reactions (list): a sequence of :class:`Reaction`.
        """
        self.reactions = reactions

    @classmethod
    def from_file(cls, filename):
        d = pickle_load(filename)
        logger.info(
            "{} reactions loaded from file: {}".format(len(d["reactions"]), filename)
        )
        return cls(d["reactions"])

    def filter_reactions_by_bond_type(self, bond_type):
        """
        Filter the reactions by the type of the breaking bond, and only reactions with the
        specified bond_type will be retained.

        Args:
            bond_type (tuple of string): species of the two atoms the bond connecting to
        """
        reactions = []
        for rxn in self.reactions:
            attr = rxn.get_broken_bond_attr()
            species = attr["species"]
            if set(species) == set(bond_type):
                reactions.append(rxn)

        self.reactions = reactions

    def group_by_reactant(self):
        """
        Group reactions that have the same reactant together.

        Returns:
            dict: with reactant as the key and list of :class:`Reaction` as the value
        """
        grouped_reactions = defaultdict(list)
        for rxn in self.reactions:
            reactant = rxn.reactants[0]
            grouped_reactions[reactant].append(rxn)
        return grouped_reactions

    def group_by_reactant_charge_0(self):
        """
        Group reactions that have the same reactant together, keeping charge 0
        reactions (charges of reactant and products are all 0).

        A group of reactions of the same reactant are put in to
        :class:`ReactionsOnePerBond` container.

        Returns:
            list: a sequence of :class:`ReactionsOnePerBond`
        """
        groups = self.group_by_reactant()

        new_groups = []
        for reactant in groups:

            zero_charge_rxns = []
            for rxn in groups[reactant]:
                zero_charge = True
                for m in rxn.reactants + rxn.products:
                    if m.charge != 0:
                        zero_charge = False
                        break
                if zero_charge:
                    zero_charge_rxns.append(rxn)

            # add to new group only when at least has one reaction
            if zero_charge_rxns:
                ropb = ReactionsOnePerBond(reactant, zero_charge_rxns)
                new_groups.append(ropb)

        return new_groups

    def group_by_reactant_lowest_energy(self):
        """
        Group reactions that have the same reactant together.

        For reactions that have the same reactant and breaks the same bond, we keep the
        reaction that have the lowest energy across products charge.

        A group of reactions of the same reactant are put in to
        :class:`ReactionsOnePerBond` container.

        Returns:
            list: a sequence of :class:`ReactionsOnePerBond`
        """

        groups = self.group_by_reactant()

        new_groups = []
        for reactant in groups:

            # find the lowest energy reaction for each bond
            lowest_energy_reaction = dict()
            for rxn in groups[reactant]:
                bond = rxn.get_broken_bond()
                if bond not in lowest_energy_reaction:
                    lowest_energy_reaction[bond] = rxn
                else:
                    e_old = lowest_energy_reaction[bond].get_free_energy()
                    e_new = rxn.get_free_energy()
                    if e_new < e_old:
                        lowest_energy_reaction[bond] = rxn

            ropb = ReactionsOnePerBond(reactant, lowest_energy_reaction.keys())
            new_groups.append(ropb)

        return new_groups

    def group_by_reactant_all(self):
        """
        Group reactions that have the same reactant together.

        A group of reactions of the same reactant are put in to
        :class:`ReactionsMultiplePerBond` container.

        Returns:
            list: a sequence of :class:`ReactionsMultiplePerBond`
        """

        groups = self.group_by_reactant()
        new_groups = [
            ReactionsMultiplePerBond(reactant, rxns) for reactant, rxns in groups.items()
        ]

        return new_groups

    def group_by_reactant_charge_pair(self):
        """
        Group reactions whose reactants are isomorphic to each other but have
        different charges.

        Then create pairs of reactions where the reactant and products of one reaction is
        is isomorphic to those of the other reaction in a pair. The pair is indexed by
        the charges of the reactants of the pair.

        Returns:
            A dict with a type (charge1, charge2) as the key, and a list of tuples as
            the value, where each tuple are two reactions (reaction1, reactions2) that
            have the same breaking bond.
        """

        grouped_reactions = self.group_by_reactant_lowest_energy()

        # groups is a list of list, where the elements of each inner list are
        # ReactionsOnePerBond instances and the corresponding reactants are
        # isomorphic to each other
        groups = []
        for rsr in grouped_reactions:
            find_iso = False
            for g in groups:
                old_rsr = g[0]
                # add to the isomorphic group
                if rsr.reactant.mol_graph.isomorphic_to(old_rsr.reactant.mol_graph):
                    g.append(rsr)
                    find_iso = True
                    break
            if not find_iso:
                g = [rsr]
                groups.append(g)

        # group by charge of a pair of reactants
        result = defaultdict(list)
        for g in groups:
            for rsr1, rsr2 in itertools.combinations(g, 2):
                if rsr2.reactant.charge < rsr1.reactant.charge:
                    rsr1, rsr2 = rsr2, rsr1
                rxn1 = {r.get_broken_bond(): r for r in rsr1.reactions}
                rxn2 = {r.get_broken_bond(): r for r in rsr2.reactions}
                res = get_same_bond_breaking_reactions_between_two_reaction_groups(
                    rsr1.reactant, rxn1, rsr2.reactant, rxn2
                )
                result[(rsr1.reactant.charge, rsr2.reactant.charge)].extend(res)
        return result

    def write_bond_energies(self, filename):
        """
        Write bond energy to a yaml file.
        """

        groups = self.group_by_reactant_all()

        # convert to nested dict: new_groups[reactant_idx][bond][charge] = rxn
        new_groups = OrderedDict()
        for rmb in groups:
            m = rmb.reactant
            key = "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
            new_groups[key] = OrderedDict()
            rsbs = rmb.group_by_bond()
            for rsb in rsbs:
                bond = rsb.broken_bond
                new_groups[key][bond] = OrderedDict()
                for rxn in rsb.reactions:
                    charge = tuple([m.charge for m in rxn.products])
                    new_groups[key][bond][charge] = rxn.as_dict()

        yaml_dump(new_groups, filename)

    def create_struct_label_dataset_reaction_network_based_classification(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        group_mode="all",
        top_n=2,
        complement_reactions=False,
        one_per_iso_bond_group=True,
    ):
        """
        Write the reaction.

        This is based on reaction network:

        1) each molecule is represented once
        2) each reaction uses the molecule index for construction instead of molecule
            instance.

        Args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            top_n (int): the top n reactions with smallest energies are categorized as
                the same class (calss 1), reactions with higher energies another class
                (class 0), and reactions without energies another class (class 2).
                If `top_n=None`, a different method to assign class is used: reactions
                with energies is categorized as class 1 and reactions without energies
                as class 0.
            complement_reactions (bool): whether to extract complement reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.

        """

        # check arguments compatibility
        if top_n is None and not complement_reactions:
            raise ValueError(
                "complement_reactions (False) should be `True` when top_n is set "
                "to None"
            )

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        # all molecules in existing reactions
        reactions = np.concatenate([grp.reactions for grp in grouped_rxns])
        mol_reservoir = set(get_molecules_from_reactions(reactions))

        # remove iso bond reactions based on one_per_iso_bond_group
        reactions = []
        for grp in grouped_rxns:
            rxns = grp.order_reactions(
                one_per_iso_bond_group, complement_reactions, mol_reservoir
            )
            reactions.extend(rxns)

        # all molecules in existing (and complementary) reactions
        # note, mol_reservoir is updated in calling grp.order_reactions
        mol_reservoir = sorted(mol_reservoir, key=lambda m: m.id)
        mol_id_to_index_mapping = {m.id: i for i, m in enumerate(mol_reservoir)}

        # use multiprocessing to get atom mappings since they are relatively expensive
        # mappings = [get_atom_bond_mapping(r) for r in reactions]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            mappings = p.map(get_atom_bond_mapping, reactions)

        all_labels = []  # one per reaction

        for i, (rxn, mps) in enumerate(zip(reactions, mappings)):

            energy = rxn.get_free_energy()

            # determine class of each reaction
            if top_n is not None:
                if energy is None:
                    cls = 2
                elif i < top_n:
                    cls = 1
                else:
                    cls = 0
            else:
                if energy is None:
                    cls = 0
                else:
                    cls = 1

            # change to index (in mol_reservoir) representation
            reactant_ids = [mol_id_to_index_mapping[m.id] for m in rxn.reactants]
            product_ids = [mol_id_to_index_mapping[m.id] for m in rxn.products]

            # bond mapping between product sdf and reactant sdf
            data = {
                "value": cls,
                "reactants": reactant_ids,
                "products": product_ids,
                "atom_mapping": mps[0],
                "bond_mapping": mps[1],
                "id": rxn.get_id(),
                "index": i,
            }
            all_labels.append(data)

        # write sdf
        self.write_sdf(mol_reservoir, struct_file)

        # label file
        yaml_dump(all_labels, label_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(mol_reservoir, bond_indices=None)
            yaml_dump(features, feature_file)

    def create_struct_label_dataset_reaction_network_based_regression(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        group_mode="all",
        one_per_iso_bond_group=True,
        write_to_file=True,
    ):
        """
        Write the reaction

        This is based on reaction network:

        1) each molecule is represented once
        2) each reaction uses the molecule index for construction instead of molecule
            instance.

        Also, this is based on the bond energy, i.e. each bond (that we have energies)
        will have one line in the label file.

        Args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.
            write_to_file (bool): if True, the results are written to files. If False,
                not write.

        Returns:
            rdkit_mols (list): rdkit molecules participating the reactions
            labels (list): each element is a dict representing a reaction
            features (list): each element is a dict representing features of a
                molecule. The size should be the same of rdkit_mols.
        """

        logger.info("Start creating struct label feature files for rxn ntwk regression")

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        # all molecules in existing reactions
        reactions = np.concatenate([grp.reactions for grp in grouped_rxns])
        mol_reservoir = set(get_molecules_from_reactions(reactions))

        # remove iso bond reactions based on one_per_iso_bond_group
        reactions = []
        for grp in grouped_rxns:
            rxns = grp.order_reactions(
                one_per_iso_bond_group,
                complement_reactions=False,
                mol_reservoir=mol_reservoir,
            )
            reactions.extend(rxns)

        # all molecules in existing (and complementary) reactions
        # note, mol_reservoir is updated in calling grp.order_reactions
        mol_reservoir = sorted(mol_reservoir, key=lambda m: m.id)
        rdkit_mols = [m.rdkit_mol for m in mol_reservoir]
        mol_id_to_index_mapping = {m.id: i for i, m in enumerate(mol_reservoir)}

        # use multiprocessing to get atom mappings since they are relatively expensive
        # mappings = [get_atom_bond_mapping(r) for r in reactions]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            mappings = p.map(get_atom_bond_mapping, reactions)

        labels = []  # one per reaction
        for i, (rxn, mps) in enumerate(zip(reactions, mappings)):

            # change to index (in mol_reservoir) representation
            reactant_ids = [mol_id_to_index_mapping[m.id] for m in rxn.reactants]
            product_ids = [mol_id_to_index_mapping[m.id] for m in rxn.products]

            # bond mapping between product sdf and reactant sdf
            data = {
                "value": rxn.get_free_energy(),
                "reactants": reactant_ids,
                "products": product_ids,
                "atom_mapping": mps[0],
                "bond_mapping": mps[1],
                "id": rxn.get_id(),
                "index": i,
            }
            labels.append(data)

        # molecule features
        features = self.get_feature(mol_reservoir, bond_indices=None)

        if write_to_file:
            # write sdf
            self.write_sdf(mol_reservoir, struct_file)

            # label file
            yaml_dump(labels, label_file)

            # write feature
            if feature_file is not None:
                yaml_dump(features, feature_file)

        logger.info("Finish creating struct label feature files for rxn ntwk regression")

        return rdkit_mols, labels, features

    def create_regression_dataset_reaction_network_simple(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        write_to_file=True,
    ):
        """
        Convert the reaction to standard data format that the fitting code expects.

        This is a simplified version of
        `create_struct_label_dataset_reaction_network_based_regression`.

        Here, will not group and order reactions and remove duplicate (reactions by
        breaking isomorphic bond in a molecule). We simply convert a list of reactions
        into the data.

        Args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str or None): filename for the feature file, if `None`,
                do not write it
            write_to_file (bool): if True, the results are written to files. If False,
                not write to files and `struct_file`, `label_file` and `feature_file`
                are ignored.

        Returns:
            rdkit_mols (list): rdkit molecules participating the reactions
            labels (list): each element is a dict representing a reaction
            features (list): each element is a dict representing features of a
                molecule. The size should be the same of rdkit_mols.
        """
        logger.info("Start creating struct label feature files for rxn ntwk regression")

        # all molecules in existing reactions
        reactions = self.reactions
        mol_reservoir = get_molecules_from_reactions(reactions)
        mol_reservoir = sorted(mol_reservoir, key=lambda m: m.id)
        mol_id_to_index_mapping = {m.id: i for i, m in enumerate(mol_reservoir)}
        rdkit_mols = [m.rdkit_mol for m in mol_reservoir]

        # use multiprocessing to get atom mappings since they are relatively expensive
        # mappings = [get_atom_bond_mapping(r) for r in reactions]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            mappings = p.map(get_atom_bond_mapping, reactions)

        labels = []  # one per reaction
        for i, (rxn, mps) in enumerate(zip(reactions, mappings)):

            # change to index (in mol_reservoir) representation
            reactant_ids = [mol_id_to_index_mapping[m.id] for m in rxn.reactants]
            product_ids = [mol_id_to_index_mapping[m.id] for m in rxn.products]

            # bond mapping between product sdf and reactant sdf
            data = {
                "value": rxn.get_free_energy(),
                "reactants": reactant_ids,
                "products": product_ids,
                "atom_mapping": mps[0],
                "bond_mapping": mps[1],
                "id": rxn.get_id(),
                "index": i,
            }
            labels.append(data)

        # features
        features = self.get_feature(mol_reservoir, bond_indices=None)

        if write_to_file:
            # write sdf
            self.write_sdf(mol_reservoir, struct_file)

            # label file
            yaml_dump(labels, label_file)

            # write feature
            if feature_file is not None:
                yaml_dump(features, feature_file)

        logger.info("Finish creating struct label feature files for rxn ntwk regression")

        return rdkit_mols, labels, features

    def create_struct_label_dataset_reaction_based_classification(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        group_mode="all",
        top_n=2,
        complement_reactions=False,
        one_per_iso_bond_group=True,
    ):
        """
        Write the reaction

        This is based on reaction:

        Each reaction uses molecule instances for its reactants and products. As a
        result, a molecule is represented multiple times, which takes long time.

        Args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            top_n (int): the top n reactions with smallest energies are categorized as
                the same class (calss 1), reactions with higher energies another class
                (class 0), and reactions without energies another class (class 2).
                If `top_n=None`, a different method to assign class is used: reactions
                with energies is categorized as class 1 and reactions without energies
                as class 0.
            complement_reactions (bool): whether to extract complement reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.

        """

        # check arguments compatibility
        if top_n is None and not complement_reactions:
            raise ValueError(
                f"complement_reactions {False} should be `True` when top_n is set "
                f"to `False`"
            )

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        all_mols = []
        all_labels = []  # one per reaction
        for grp in grouped_rxns:
            reactions, _ = grp.order_reactions(
                one_per_iso_bond_group, complement_reactions
            )

            # rxn: a reaction for one bond and a specific combination of charges
            for i, rxn in enumerate(reactions):
                mols = rxn.reactants + rxn.products
                energy = rxn.get_free_energy()

                # determine class of each reaction
                if top_n is not None:
                    if energy is None:
                        cls = 2
                    elif i < top_n:
                        cls = 1
                    else:
                        cls = 0
                else:
                    if energy is None:
                        cls = 0
                    else:
                        cls = 1

                # bond mapping between product sdf and reactant sdf
                all_mols.extend(mols)
                data = {
                    "value": cls,
                    "num_mols": len(mols),
                    "atom_mapping": rxn.atom_mapping(),
                    "bond_mapping": rxn.bond_mapping_by_sdf_int_index(),
                    "id": rxn.get_id(),
                }
                all_labels.append(data)

        # write sdf
        self.write_sdf(all_mols, struct_file)

        # label file
        yaml_dump(all_labels, label_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(all_mols, bond_indices=None)
            yaml_dump(features, feature_file)

    def create_struct_label_dataset_reaction_based_regression(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        group_mode="all",
        one_per_iso_bond_group=True,
    ):
        """
        Write the reaction

        This is based on reaction:

        Each reaction uses molecule instances for its reactants and products. As a
        result, a molecule is represented multiple times, which takes long time.

        Args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.
        """

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        all_mols = []
        all_labels = []  # one per reaction

        for grp in grouped_rxns:
            reactions, _ = grp.order_reactions(
                one_per_iso_bond_group, complement_reactions=False
            )

            # rxn: a reaction for one bond and a specific combination of charges
            for i, rxn in enumerate(reactions):
                mols = rxn.reactants + rxn.products
                energy = rxn.get_free_energy()

                # bond mapping between product sdf and reactant sdf
                all_mols.extend(mols)
                data = {
                    "value": energy,
                    "num_mols": len(mols),
                    "atom_mapping": rxn.atom_mapping(),
                    "bond_mapping": rxn.bond_mapping_by_sdf_int_index(),
                    "id": rxn.get_id(),
                }
                all_labels.append(data)

        # write sdf
        self.write_sdf(all_mols, struct_file)

        # label file
        yaml_dump(all_labels, label_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(all_mols, bond_indices=None)
            yaml_dump(features, feature_file)

    def create_struct_label_dataset_bond_based_classification(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        group_mode="charge_0",
        top_n=2,
        complement_reactions=True,
        one_per_iso_bond_group=True,
    ):
        """
        Write the reaction class to files.

        Also, this is based on the bond energy, i.e. each bond (that we have energies)
        will have one line in the label file.

        Args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            top_n (int): the top n reactions with smallest energies are categorized as
                the same class (calss 1), reactions with higher energies another class
                (class 0), and reactions without energies another class (class 2).
                If `top_n=None`, a different method to assign class is used: reactions
                with energies is categorized as class 1 and reactions without energies
                as class 0
            complement_reactions (bool): whether to extract complement reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.
        """

        def write_label(reactants, bond_idx, label_class, filename="label.txt"):
            """
            Write bond energy class to file.

            See the text below on how the info is written.

            Args:
                reactants (list ): MoleculeWrapper objects
                bond_idx (list of int): the index of the broken bond in the reactant;
                filename (str): name of the file to write the label
            """

            create_directory(filename)
            with open(to_path(filename), "w") as f:
                f.write(
                    "# Each line lists the energy class of a bond in a molecule. "
                    "Each line has three items: "
                    "1st: an integer of of {0,1,2}, indicating the class of bond energy, "
                    "0 stands for feasible reaction, 1 stands for nonfeasize reaction "
                    "and 2 stands for unknown, i.e. we do not have info about the "
                    "reaction."
                    "2nd: index of the bond in the molecule {0,1,2,num_bonds-1}."
                    "3rd: molecule idx from which the bond come.\n"
                )

                for i, (m, idx, lb) in enumerate(zip(reactants, bond_idx, label_class)):
                    f.write("{} {} {}\n".format(lb, idx, m.id))

        # check arguments compatibility
        if top_n is None and not complement_reactions:
            raise ValueError(
                f"complement_reactions {False} should be `True` when top_n is set "
                f"to `None`"
            )

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        all_reactants = []
        broken_bond_idx = []  # int index in sdf molecule
        broken_bond_pairs = []  # a tuple index in graph molecule
        label_class = []
        for grp in grouped_rxns:
            reactant = grp.reactant

            ordered_rxns = grp.order_reactions(
                complement_reactions, one_per_iso_bond_group
            )
            rxns_dict = {
                rxn.get_broken_bond(): (i, rxn) for i, rxn in enumerate(ordered_rxns)
            }

            # bond energies in the same ordering as in sdf file
            sdf_bonds = reactant.get_sdf_bond_indices(zero_based=True)
            for ib, bond in enumerate(sdf_bonds):

                # when one_per_iso_bond_group is `True`, some bonds are deleted
                if bond not in rxns_dict:
                    continue

                i, rxn = rxns_dict[bond]
                energy = rxn.get_free_energy()

                # determine class of each reaction
                if top_n is not None:
                    if energy is None:
                        cls = 2
                    elif i < top_n:
                        cls = 1
                    else:
                        cls = 0
                else:
                    if energy is None:
                        cls = 0
                    else:
                        cls = 1

                all_reactants.append(reactant)
                broken_bond_idx.append(ib)
                broken_bond_pairs.append(bond)
                label_class.append(cls)

        # write label
        write_label(all_reactants, broken_bond_idx, label_class, label_file)

        # write sdf
        self.write_sdf(all_reactants, struct_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(all_reactants, bond_indices=broken_bond_pairs)
            yaml_dump(features, feature_file)

    def create_struct_label_dataset_bond_based_regression(
        self,
        struct_file="sturct.sdf",
        label_file="label.yaml",
        feature_file=None,
        group_mode="charge_0",
        one_per_iso_bond_group=True,
    ):
        """
        Write the reactions to files.

        Also, this is based on the bond energy, i.e. each bond (that we have energies)
        will have one line in the label file.

        args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            group_mode (str): the method to group reactions, different mode result in
                different reactions to be retained, e.g. `charge_0` keeps all charge 0
                reactions.
            one_per_iso_bond_group (bool): whether to keep just one reaction from each
                iso bond group.

        """

        def get_labels(reactions, bond_idx):
            labels = []

            for i, (rxn, idx) in enumerate(zip(reactions, bond_idx)):
                reactant = rxn.reactants[0]
                num_bonds = len(reactant.bonds)
                labels.append(
                    {
                        "value": [rxn.get_free_energy()],
                        "bond_index": [idx],
                        "molecule_id": reactant.id,
                        "num_bonds_in_molecule": num_bonds,
                        "reaction_id": rxn.get_id(),
                    }
                )

            return labels

        if group_mode == "all":
            grouped_rxns = self.group_by_reactant_all()
        elif group_mode == "charge_0":
            grouped_rxns = self.group_by_reactant_charge_0()
        elif group_mode == "energy_lowest":
            grouped_rxns = self.group_by_reactant_lowest_energy()
        else:
            raise ValueError(
                f"group_mode ({group_mode}) not supported. Options are: 'all', "
                f"'charge_0', and 'energy_lowest'."
            )

        all_rxns = []
        broken_bond_idx = []
        broken_bond_pairs = []
        for grp in grouped_rxns:
            reactant = grp.reactant

            ordered_rxns = grp.order_reactions(
                one_per_iso_bond_group, complement_reactions=False
            )
            rxns_dict = {
                rxn.get_broken_bond(): (i, rxn) for i, rxn in enumerate(ordered_rxns)
            }

            # bond energies in the same order as in sdf file
            sdf_bonds = reactant.get_sdf_bond_indices(zero_based=True)
            for ib, bond in enumerate(sdf_bonds):

                # when one_per_iso_bond_group is `True`, some bonds are deleted
                if bond not in rxns_dict:
                    continue

                _, rxn = rxns_dict[bond]

                all_rxns.append(rxn)
                broken_bond_idx.append(ib)
                broken_bond_pairs.append(bond)

        all_reactants = [rxn.reactants[0] for rxn in all_rxns]

        # write label
        labels = get_labels(all_rxns, broken_bond_idx)
        yaml_dump(labels, label_file)

        # write sdf
        self.write_sdf(all_reactants, struct_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(all_reactants, bond_indices=broken_bond_pairs)
            yaml_dump(features, feature_file)

    def create_struct_label_dataset_mol_based(
        self,
        struct_file="sturct.sdf",
        label_file="label.txt",
        feature_file=None,
        lowest_across_product_charge=True,
    ):
        """
        Write the reactions to files.

        The is molecule based, each molecule will have a line in the label file.

        args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
            lowest_across_product_charge (bool): If `True` each reactant corresponds to
                the lowest energy products. If `False`, find all 0->0+0 reactions,
                i.e. the charge of reactant and products should all be zero.

        """
        if lowest_across_product_charge:
            grouped_reactions = self.group_by_reactant_lowest_energy()
        else:
            grouped_reactions = self.group_by_reactant_charge_0()

        # write label
        create_directory(label_file)
        with open(to_path(label_file), "w") as f:
            f.write(
                "# Each line lists the bond energies of a molecule. "
                "The number of items in each line is equal to 2*N, where N is the "
                "number bonds. The first N items are bond energies and the next N "
                "items are indicators (0 or 1) to specify whether the bond energy "
                "exists in the dataset. A value of 0 means the corresponding bond "
                "energy should be ignored, whatever its value is.\n"
            )

            for rsr in grouped_reactions:
                reactant = rsr.reactant

                # get a mapping between sdf bond and reactions
                rxns_by_sdf_bond = dict()
                for rxn in rsr.reactions:
                    bond = rxn.get_broken_bond()
                    rxns_by_sdf_bond[bond] = rxn

                # write bond energies in the same order as sdf file
                energy = []
                indicator = []
                sdf_bonds = reactant.get_sdf_bond_indices(zero_based=True)
                for ib, bond in enumerate(sdf_bonds):
                    # have reaction that breaks this bond
                    if bond in rxns_by_sdf_bond:
                        rxn = rxns_by_sdf_bond[bond]
                        energy.append(rxn.get_free_energy())
                        indicator.append(1)
                    else:
                        energy.append(0.0)
                        indicator.append(0)

                for i in energy:
                    f.write("{:.15g} ".format(i))
                f.write("    ")
                for i in indicator:
                    f.write("{} ".format(i))
                f.write("\n")

        # write sdf
        reactants = [rsr.reactant for rsr in grouped_reactions]
        self.write_sdf(reactants, struct_file)

        # write feature
        if feature_file is not None:
            features = self.get_feature(reactants, bond_indices=None)
            yaml_dump(features, feature_file)

    @staticmethod
    def write_sdf(molecules, filename="struct.sdf"):
        """
        Write molecules to sdf.

        Args:
            molecules (list): a sequence of :class:`MoleculeWrapper`
            filename (str): output filename
        """
        logger.info("Start writing sdf file: {}".format(filename))

        create_directory(filename)
        with open(to_path(filename), "w") as f:
            for i, m in enumerate(molecules):
                name = "{}_{}_{}_{}_index-{}".format(
                    m.id, m.formula, m.charge, m.free_energy, i
                )
                sdf = m.write(name=name)
                f.write(sdf)

        logger.info("Finish writing sdf file: {}".format(filename))

    @staticmethod
    def get_feature(molecules, bond_indices=None):
        """
        Get features from molecule.

        Args:
            molecules (list): a sequence of :class:`MoleculeWrapper`
            bond_indices (list of tuple or None): broken bond in the corresponding
                molecule

        Returns:
            list: features extracted from wrapper molecule, a dict for each molecule.
        """

        all_feats = []
        for i, m in enumerate(molecules):
            if bond_indices is None:
                idx = None
            else:
                idx = bond_indices[i]
            feat = m.pack_features(broken_bond=idx)
            if "index" not in feat:
                feat["index"] = i
            all_feats.append(feat)

        return all_feats


def get_molecules_from_reactions(reactions):
    """Return a list of unique molecules participating in all reactions."""
    mols = set()
    for r in reactions:
        mols.update(r.reactants + r.products)
    return list(mols)


def get_atom_bond_mapping(rxn):
    atom_mp = rxn.atom_mapping()
    bond_mp = rxn.bond_mapping_by_sdf_int_index()
    return atom_mp, bond_mp


def get_same_bond_breaking_reactions_between_two_reaction_groups(
    reactant1, group1, reactant2, group2
):
    """
    Args:
        reactant1 (MolecularWrapper instance)
        group1, (dict): A group of reactions that have the same reactant1 but
            breaking different bonds. the bond indices is the key of the dict.
        reactant2 (MolecularWrapper instance) reactant2 should have the same
            isomorphism as that of reactant1, but other property can be different,
            e.g. (charge).
        group2 (dict): A group of reactions that have the same reactant2 but
            breaking different bonds. the bond indices is the key of the dict.

    Returns:
        A list of tuples (rxn1, rxn2) where rxn1 and rxn2 has the same breaking bond.
    """

    bonds1 = [tuple(k) for k in group1]
    bonds2 = [tuple(k) for k in group2]
    fragments1 = reactant1.fragments[bonds1]
    fragments2 = reactant2.fragments[bonds2]

    res = []
    for b1, mgs1 in fragments1.items():
        for b2, mgs2 in fragments2.items():
            if len(mgs1) == len(mgs2) == 1:
                if mgs1[0].isomorphic_to(mgs2[0]):
                    res.append((group1[b1], group2[b2]))

            if len(mgs1) == len(mgs2) == 2:
                if (
                    mgs1[0].isomorphic_to(mgs2[0]) and mgs1[1].isomorphic_to(mgs2[1])
                ) or (mgs1[0].isomorphic_to(mgs2[1]) and mgs1[1].isomorphic_to(mgs2[0])):
                    res.append((group1[b1], group2[b2]))
    return res
