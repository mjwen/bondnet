import itertools
import logging
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.isomorphism as iso
from collections import defaultdict, OrderedDict
from gnn.data.database import DatabaseOperation
from gnn.utils import (
    create_directory,
    pickle_dump,
    pickle_load,
    yaml_dump,
    yaml_load,
    expand_path,
)

logger = logging.getLogger(__name__)


class Reaction:
    """
    A reaction that only has one bond break or the type ``A -> B + C``
    (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

    Args:
        reactants: a list of Molecules
        products: a list of Molecules
        broken_bond (tuple): index indicating the broken bond in the mol_graph
    """

    # NOTE most methods in this class only works for A->B and A->B+C type reactions

    def __init__(self, reactants, products, broken_bond=None):
        assert len(reactants) == 1, "incorrect number of reactants, should be 1"
        assert 1 <= len(products) <= 2, "incorrect number of products, should be 1 or 2"
        self.reactants = reactants
        # ordered products needed by `group_by_reactant_bond_and_charge(self)`
        # where we use charge as a dict key
        self.products = self._order_molecules(products)
        self.broken_bond = broken_bond

    def get_broken_bond(self):
        if self.broken_bond is None:
            if len(self.products) == 1:
                bond = is_valid_A_to_B_reaction(self.reactants[0], self.products[0])
            else:
                bond = is_valid_A_to_B_C_reaction(self.reactants[0], self.products)
            if bond is None:
                raise RuntimeError(
                    "invalid reaction (cannot break a reactant bond to get products)"
                )
            self.broken_bond = bond

        return self.broken_bond

    def get_broken_bond_attr(self):
        """
        Returns a dict of the species and bond order of the broken bond.
        """
        graph = self.reactants[0].graph
        u, v = self.get_broken_bond()
        spec_u = graph.nodes[u]["specie"]
        spec_v = graph.nodes[v]["specie"]
        # TODO, we temporarily need the try except block because the
        #  metal_edge_extender does not add weight/get
        try:
            order = graph.get_edge_data(u, v, key=0)[
                "weight"
            ]  # key=0 because of MultiGraph
        except KeyError:
            order = 0
        return {"species": sorted([spec_u, spec_v]), "order": order}

    def get_reaction_free_energy(self):
        energy = 0
        for mol in self.reactants:
            energy -= mol.free_energy
        for mol in self.products:
            energy += mol.free_energy
        return energy

    def pack_features(self):
        """
        Prepare the features that may be used by ML model, e.g. partial charge on
        subgraphs after a bond breaking.

        Returns:
            dict of features.
        """

        reactant = self.reactants[0]
        broken_bond = self.get_broken_bond()
        mappings = reactant.subgraph_atom_mapping(broken_bond)
        resp = []
        mulliken = []
        atom_spin = []
        for m in mappings:
            resp.append([reactant.resp[i] for i in m])
            mulliken.append([reactant.mulliken[i] for i in m])
            atom_spin.append([reactant.atom_spin[i] for i in m])

        feats = dict()
        feats["abs_resp_diff"] = abs(sum(resp[0]) - sum(resp[1]))
        feats["abs_mulliken_diff"] = abs(sum(mulliken[0]) - sum(mulliken[1]))
        feats["abs_atom_spin_diff"] = abs(sum(atom_spin[0]) - sum(atom_spin[1]))

        return feats

    def as_dict(self):
        d = {
            "reactants": [
                "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
                for m in self.reactants
            ],
            "products": [
                "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
                for m in self.products
            ],
            "charge": [m.charge for m in self.reactants + self.products],
            "broken_bond": self.get_broken_bond(),
            "bond_energy": self.get_reaction_free_energy(),
        }
        return d

    def to_file(self, filename):
        mols = self.reactants + self.products
        for m in mols:
            m.make_picklable()
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": self.reactants,
            "products": self.products,
            "broken_bond": self.broken_bond,
        }
        pickle_dump(d, filename)

    @classmethod
    def from_file(cls, filename):
        d = pickle_load(filename)
        return cls(d["reactants"], d["products"], d["broken_bond"])

    @staticmethod
    def _order_molecules(mols):
        """
        Order the molecules according to the below rules (in order):
        1. molecular mass
        2. number of atoms
        3. number of bonds
        4. alphabetical formula
        5. diameter of molecule graph, i.e. largest distance for node to node
        6. charge

        Args:
            mols: a list of molecules.

        Returns:
            A list of ordered molecules.
        """

        def compare(pa, pb, a, b):
            if pa < pb:
                return [a, b]
            elif pa > pb:
                return [b, a]
            else:
                return None

        def order_by_weight(a, b):
            pa = a.weight
            pb = b.weight
            return compare(pa, pb, a, b)

        def order_by_natoms(a, b):
            pa = len(a.atoms)
            pb = len(b.atoms)
            return compare(pa, pb, a, b)

        def order_by_nbonds(a, b):
            pa = len(a.bonds)
            pb = len(b.bonds)
            return compare(pa, pb, a, b)

        def order_by_formula(a, b):
            pa = a.formula
            pb = b.formula
            return compare(pa, pb, a, b)

        def order_by_diameter(a, b):
            try:
                pa = nx.diameter(a.graph)
            except nx.NetworkXError:
                pa = 100000000
            try:
                pb = nx.diameter(b.graph)
            except nx.NetworkXError:
                pb = 100000
            return compare(pa, pb, a, b)

        def order_by_charge(a, b):
            pa = a.charge
            pb = b.charge
            return compare(pa, pb, a, b)

        if len(mols) == 1:
            return mols

        out = order_by_weight(*mols)
        if out is not None:
            return out
        out = order_by_natoms(*mols)
        if out is not None:
            return out
        out = order_by_nbonds(*mols)
        if out is not None:
            return out
        out = order_by_formula(*mols)
        if out is not None:
            return out
        out = order_by_diameter(*mols)
        if out is not None:
            return out
        a, b = mols
        if a.mol_graph.isomorphic_to(b.mol_graph):
            out = order_by_charge(*mols)  # e.g. H+ and H-
            if out is not None:
                return out
            else:
                return mols  # two exactly the same molecules
        raise RuntimeError("Cannot order molecules")


class ReactionExtractor:
    def __init__(self, molecules, reactions=None):
        self.molecules = molecules or self._get_molecules_from_reactions(reactions)
        self.reactions = reactions

        self.buckets = None
        self.bucket_keys = None

    # def get_molecule_properties(self, keys):
    #     values = defaultdict(list)
    #     for m in self.molecules:
    #         for k in keys:
    #             values[k].append(getattr(m, k))
    #     return values

    def bucket_molecules(self, keys=["formula", "charge"]):
        """
        Classify molecules into nested dictionaries according to molecule properties
        specified in ``keys``.

        Args:
            keys (list of str): each str should be a molecule property.

        Returns:
            nested dictionary of molecules classified according to keys.
        """
        logger.info("Start bucketing molecules...")

        num_keys = len(keys)
        buckets = {}
        for m in self.molecules:
            b = buckets
            for i, k in enumerate(keys):
                v = getattr(m, k)
                if i == num_keys - 1:
                    b.setdefault(v, []).append(m)
                else:
                    b.setdefault(v, {})
                b = b[v]

        self.bucket_keys = keys
        self.buckets = buckets

    def extract_A_to_B_style_reaction(self):
        """
        Return a list of A -> B reactions.
        """
        logger.info("Start extracting A -> B style reactions")

        if self.buckets is None:
            self.bucket_molecules(keys=["formula", "charge"])

        A2B = []
        i = 0
        for _, entries_formula in self.buckets.items():
            i += 1
            if i % 10000 == 0:
                print("@@flag A->B running bucket", i)
            for _, entries_charges in entries_formula.items():
                for A, B in itertools.permutations(entries_charges, 2):
                    bond = is_valid_A_to_B_reaction(A, B)
                    if bond is not None:
                        A2B.append(Reaction([A], [B], bond))
        self.reactions = A2B

        return A2B

    def extract_A_to_B_C_style_reaction(self):
        """
        Return a list of A -> B + C reactions.
        """
        logger.info("Start extracting A -> B + C style reactions")

        if self.buckets is None:
            self.bucket_molecules(keys=["formula", "charge"])

        fcmap = self._get_formula_composition_map(self.molecules)
        A2BC = []
        i = 0
        for (
            (formula_A, entries_formula_A),
            (formula_B, entries_formula_B),
            (formula_C, entries_formula_C),
        ) in itertools.product(self.buckets.items(), repeat=3):

            i += 1
            if i % 10000 == 0:
                print("@@flag A->B+C running bucket", i)

            composition_A = fcmap[formula_A]
            composition_B = fcmap[formula_B]
            composition_C = fcmap[formula_C]

            if not self._is_valid_A_to_B_C_composition(
                composition_A, composition_B, composition_C
            ):
                continue

            reaction_ids = []
            for (
                (charge_A, entries_charge_A),
                (charge_B, entries_charge_B),
                (charge_C, entries_charge_C),
            ) in itertools.product(
                entries_formula_A.items(),
                entries_formula_B.items(),
                entries_formula_C.items(),
            ):
                if not self._is_valid_A_to_B_C_charge(charge_A, charge_B, charge_C):
                    continue

                for A, B, C in itertools.product(
                    entries_charge_A, entries_charge_B, entries_charge_C
                ):
                    bond = is_valid_A_to_B_C_reaction(A, [B, C])
                    if bond is not None:
                        ids = {A.id, B.id, C.id}
                        # remove repeating reactions (e.g. A->B+C and A->C+B)
                        if ids not in reaction_ids:
                            A2BC.append(Reaction([A], [B, C], bond))
                            reaction_ids.append(ids)

        self.reactions = A2BC

        return A2BC

    def extract_one_bond_break(self):
        """
        Extract all reactions that only has one bond break or the type ``A -> B + C``
        (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

        Returns:
            A list of reactions.
        """
        A2B = self.extract_A_to_B_style_reaction()
        A2BC = self.extract_A_to_B_C_style_reaction()
        self.reactions = A2B + A2BC

        return A2B, A2BC

    def filter_reactions_by_reactant_charge(self, charge):
        """
        Filter the reactions by the charge of the reactant, and only reactions with the
        specified charge will be retained.

        Args:
            charge (int): charge of reactant of the reactions to filter on.
        """
        reactions = []
        for rxn in self.reactions:
            if rxn.reactants[0].charge == charge:
                reactions.append(rxn)
        self.reactions = reactions

    def filter_reactions_by_bond_type_and_order(self, bond_type, bond_order=None):
        """
        Filter the reactions by the type of the breaking bond, and only reactions with the
        specified bond_type will be retained.

        Args:
            bond_type (tuple of string): species of the two atoms the bond connecting to
            bond_order (int): bond order to filter on
        """
        reactions = []
        for rxn in self.reactions:
            attr = rxn.get_broken_bond_attr()
            species = set(attr["species"])
            order = attr["order"]
            if set(species) == set(bond_type):
                if bond_order is None:
                    reactions.append(rxn)
                else:
                    if order == bond_order:
                        reactions.append(rxn)

        self.reactions = reactions

    def group_by_reactant(self):
        """
        Return: a dict with reactant as the key and list of reactions as the value.
        """
        grouped_reactions = defaultdict(list)
        for rxn in self.reactions:
            reactant = rxn.reactants[0]
            grouped_reactions[reactant].append(rxn)
        return grouped_reactions

    def group_by_reactant_bond_and_charge(self):
        """
        Group all the reactions to nested dicts according to the reactant, bond, and
        charge.

        Returns:
            A dict of dict of dict. The outer dict has reactant index as the key,
            the middle dict has bond indices (a tuple) as the key and the inner dict
            has charges (a tuple) as the key and bond attributes (a dict of energy,
            bond order, ect.). as the value.
        """
        grouped_reactions = self.group_by_reactant()

        groups = OrderedDict()
        for reactant, reactions in grouped_reactions.items():
            groups[reactant] = OrderedDict()

            for bond in reactant.graph.edges():
                groups[reactant][bond] = dict()

            for rxn in reactions:
                charge = tuple([m.charge for m in rxn.reactants + rxn.products])
                bond = rxn.get_broken_bond()
                groups[reactant][bond][charge] = rxn.as_dict()

        return groups

    def group_by_reactant_bond_keep_lowest_energy_across_products_charge(self):
        """
        Group all the reactions to nested dicts according to the reactant and bond.
        For cases where products have different charges, we keep the reaction with the
        lowess energy.

        Returns:
            A dict of dict. The outer dict has reactant as the key and the inner dict has
            bond (a tuple) as the key and Reaction instance as the value.
        """
        grouped_reactions = self.group_by_reactant()

        groups = OrderedDict()
        for reactant, reactions in grouped_reactions.items():
            groups[reactant] = OrderedDict()

            for rxn in reactions:
                bond = rxn.get_broken_bond()
                if bond not in groups[reactant]:
                    groups[reactant][bond] = rxn
                else:
                    e_old = groups[reactant][bond].get_reaction_free_energy()
                    e_new = rxn.get_reaction_free_energy()
                    if e_new < e_old:
                        groups[reactant][bond] = rxn

        return groups

    def group_by_reactant_charge(self):
        """
        Get the energy difference of reactions that has the same isomorphic reactant
        but different charge.
        e.g. M(+1) M(0)

        Returns:
            A dict: with a type (charge1, charge2) as the key, and a list of types of
            the value, where each tuple are two reactions (reaction1, reactions2) that
            has the same breaking bond.
        """

        grouped_reactions = (
            self.group_by_reactant_bond_keep_lowest_energy_across_products_charge()
        )

        # groups is A list of dict, where the keys of each dict are isomorphic to each
        # other.
        groups = []
        for reactant, reactions in grouped_reactions.items():
            find_iso = False
            for g in groups:
                # get the first in the isomorphic group
                for m in g:
                    iso_m = m
                    break
                # add to the isomorphic group
                if iso_m.mol_graph.isomorphic_to(reactant.mol_graph):
                    g[reactant] = reactions
                    find_iso = True
                    break

            if not find_iso:
                g = OrderedDict()
                g[reactant] = reactions
                groups.append(g)

        # group by charge of a pair of reactants
        result = defaultdict(list)
        for g in groups:
            reactants = list(g.keys())
            for r1, r2 in itertools.combinations(reactants, 2):
                if r2.charge < r1.charge:
                    r1, r2 = r2, r1

                res = get_same_bond_breaking_reactions_between_two_reaction_groups(
                    r1, g[r1], r2, g[r2]
                )
                result[(r1.charge, r2.charge)].extend(res)
        return result

    def get_reactions_with_lowest_energy(self):
        """
        Get the reactions by removing higher energy ones. Higher energy is compared
        across product charge.

        Returns:
            A list of Reaction.
        """
        groups = self.group_by_reactant_bond_keep_lowest_energy_across_products_charge()
        reactions = []
        for _, rxns in groups.items():
            for _, r in rxns.items():
                reactions.append(r)
        return reactions

    def write_bond_energies(self, filename, mode="reactant_bond_charge"):
        if mode == "reactant_bond_charge":
            groups = self.group_by_reactant_bond_and_charge()
        elif mode == "reactant_charge_bond":
            groups = self.group_by_reactant_charge_and_bond()
        else:
            raise RuntimeError("mode unsupported")
        for m in self.molecules:
            m.make_picklable()

        # change reactant (which is the key of the outer dict) to string
        new_groups = OrderedDict()
        for m, v in groups.items():
            idx = "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
            new_groups[idx] = v

        yaml_dump(new_groups, filename)

    @staticmethod
    def write_sdf(molecules, filename="molecules.sdf"):
        """
        Write molecules sdf to file.

        Args:
            filename (str): output filename
            molecules: an iterable of molecules, e.g. list, OrderedDict
        """
        logger.info("Start writing to sdf file: {}".format(filename))
        filename = expand_path(filename)
        create_directory(filename)
        with open(filename, "w") as f:
            for i, m in enumerate(molecules):
                msg = "{}_{}_{}_{} int_id: {}".format(
                    m.formula, m.charge, m.id, m.free_energy, i
                )
                sdf = m.write(file_format="sdf", message=msg)
                f.write(sdf)

    @staticmethod
    def write_feature(reactions, filename="feature.yaml"):
        """
        Write molecules features to file.

        Args:
            reactions (list): a list of reactions
            filename (str): output filename
        """
        logger.info("Start writing feature file: {}".format(filename))

        all_feats = []
        for rxn in reactions:
            m = rxn.reactants[0]
            feat = m.pack_features(use_obabel_idx=True)
            feat.update(rxn.pack_features())
            all_feats.append(feat)
        yaml_dump(all_feats, filename)

        logger.info("Finish writing feature file: {}".format(filename))

    def create_struct_label_dataset_bond_based(
        self, struct_file="sturct.sdf", label_file="label.txt", feature_file=None
    ):
        """
        Write the reactions to files.

        Also, this is based on the bond energy, i.e. each bond (that we have energies)
        will have one entry.

        args:
            struct_file (str): filename of the sdf structure file
            label_file (str): filename of the label
            feature_file (str): filename for the feature file, if `None`, do not write it
        """
        grouped_reactions = (
            self.group_by_reactant_bond_keep_lowest_energy_across_products_charge()
        )

        # write label
        all_rxns = []
        label_file = expand_path(label_file)
        create_directory(label_file)
        with open(label_file, "w") as f:
            f.write(
                "# Each line lists the bond energies of a molecule. "
                "The number of items in each line is equal to 2*N, where N is the "
                "number bonds. The first N items are bond energies and the next N "
                "items are indicators (0 or 1) to specify whether the bond energy "
                "exists in the dataset. A value of 0 means the corresponding bond "
                "energy should be ignored, whatever its value is.\n"
            )
            for reactant, reactions in grouped_reactions.items():

                current_rxns = dict()
                for bond, rxn in reactions.items():
                    bond = tuple(sorted(reactant.graph_bond_idx_to_ob_bond_idx(bond)))
                    # note we need to keep track of it because the order changes when
                    # we write it out below
                    current_rxns[bond] = rxn

                # write bond energies in the same order as sdf file
                sdf_bonds = reactant.get_sdf_bond_indices()
                for ib, bond in enumerate(sdf_bonds):

                    if bond not in current_rxns:  # do not have reaction breaking bond
                        continue

                    rxn = current_rxns[bond]
                    all_rxns.append(rxn)

                    attr = rxn.as_dict()

                    # write bond energies
                    for ii in range(len(sdf_bonds)):
                        if ii == ib:
                            f.write("{:.15g} ".format(attr["bond_energy"]))
                        else:
                            f.write("0.0 ")
                    f.write("   ")

                    # write bond energy indicator
                    for ii in range(len(sdf_bonds)):
                        if ii == ib:
                            f.write("1 ")
                        else:
                            f.write("0 ")

                    # write other info (reactant and product info, and bond energy)
                    f.write(
                        "    # {} {} {} {}\n".format(
                            attr["reactants"],
                            attr["products"],
                            reactant.graph_bond_idx_to_ob_bond_idx(attr["broken_bond"]),
                            attr["bond_energy"],
                        )
                    )

        # write sdf
        reactants = [rxn.reactants[0] for rxn in all_rxns]
        self.write_sdf(reactants, struct_file)

        # write feature
        if feature_file is not None:
            self.write_feature(all_rxns, feature_file)

    # def create_struct_label_dataset_with_lowest_energy_across_charge_bond_based(
    #     self, struct_file="sturct.sdf", label_file="label.txt", feature_file=None
    # ):
    #     """
    #     Write the reactions to files.
    #
    #     Each reactant may have multiple products corresponding to various charge
    #     combinations. Here, we write the lowest energy one.
    #
    #     Also, this is based on the bond energy, i.e. each bond (that we have energies)
    #     will have one entry.
    #
    #     args:
    #         struct_file (str): filename of the sdf structure file
    #         label_file (str): filename of the label
    #         feature_file (str): filename for the feature file, if `None`, do not write it
    #     """
    #     grouped_reactions = self.group_by_reactant_bond_and_charge()
    #
    #     # write label #     mols = []
    #     label_file = expand_path(label_file)
    #     create_directory(label_file)
    #     with open(label_file, "w") as f:
    #         f.write(
    #             "# Each line lists the molecule charge and bond energies of a molecule. "
    #             "The number of items in each line is equal to 1 + 2*N, where N is the "
    #             "number bonds. The first item is the molecule charge. The first half "
    #             "of the remaining items are bond energies and the next half values are "
    #             "indicators (0 or 1) to specify whether the bond energy exist in the "
    #             "dataset. A value of 0 means the corresponding bond energy should be "
    #             "ignored, whatever its value is.\n"
    #         )
    #         for reactant, reactions in grouped_reactions.items():
    #
    #             bonds_energy = dict()
    #             other_attr = dict()
    #             for bond, rxns in reactions.items():
    #                 bond = tuple(sorted(reactant.graph_bond_idx_to_ob_bond_idx(bond)))
    #                 bonds_energy[bond] = None
    #                 other_attr[bond] = None
    #                 for charge, attr in rxns.items():
    #                     if attr:
    #                         # find the min energy across charge
    #                         if bonds_energy[bond] is not None:
    #                             if attr["bond_energy"] < bonds_energy[bond]:
    #                                 bonds_energy[bond] = attr["bond_energy"]
    #                                 other_attr[bond] = attr
    #                         else:
    #                             bonds_energy[bond] = attr["bond_energy"]
    #                             other_attr[bond] = attr
    #
    #             for _, energy in bonds_energy.items():
    #                 if energy is not None:
    #                     mols.append(reactant)
    #
    #             # write bond energies in the same order as sdf file
    #             sdf_bonds = reactant.get_sdf_bond_indices()
    #             for ib, bond in enumerate(sdf_bonds):
    #                 energy = bonds_energy[bond]
    #                 attr = other_attr[bond]
    #
    #                 if energy is not None:
    #
    #                     # write charge
    #                     f.write("{}    ".format(reactant.charge))
    #
    #                     # write bond energies
    #                     for ii in range(len(sdf_bonds)):
    #                         if ii == ib:
    #                             f.write("{:.15g} ".format(energy))
    #                         else:
    #                             f.write("0.0 ")
    #                     f.write("   ")
    #
    #                     # write bond energy indicator
    #                     for ii in range(len(sdf_bonds)):
    #                         if ii == ib:
    #                             f.write("1 ")
    #                         else:
    #                             f.write("0 ")
    #
    #                     # write other info (reactant and product info, and bond energy)
    #                     f.write(
    #                         "    # {} {} {} {}\n".format(
    #                             attr["reactants"],
    #                             attr["products"],
    #                             reactant.graph_bond_idx_to_ob_bond_idx(
    #                                 attr["broken_bond"]
    #                             ),
    #                             attr["bond_energy"],
    #                         )
    #                     )
    #
    #     # write sdf
    #     self.write_sdf(mols, struct_file)
    #
    #     # write feature
    #     if feature_file is not None:
    #         self.write_feature(mols, feature_file)
    #
    # def create_struct_label_dataset_with_lowest_energy_across_charge(
    #     self, struct_file="sturct.sdf", label_file="label.txt", feature_file=None
    # ):
    #     """
    #     Write the reactions to files.
    #
    #     Each reactant may have multiple products corresponding to various charge
    #     combinations. Here, we write the lowest energy one.
    #
    #     args:
    #         struct_file (str): filename of the sdf structure file
    #         label_file (str): filename of the laels
    #         feature_file (str): filename for the feature file, if `None`, do not write it
    #     """
    #     grouped_reactions = self.group_by_reactant_bond_and_charge(False, True)
    #
    #     # write label
    #     label_file = expand_path(label_file)
    #     create_directory(label_file)
    #     with open(label_file, "w") as f:
    #         f.write(
    #             "# Each line lists the molecule charge and bond energies of a molecule. "
    #             "The number of items in each line is equal to 1 + 2*N, where N is the "
    #             "number bonds. The first item is the molecule charge. The first half "
    #             "of the remaining items are bond energies and the next half values are "
    #             "indicators (0 or 1) to specify whether the bond energy exist in the "
    #             "dataset. A value of 0 means the corresponding bond energy should be "
    #             "ignored, whatever its value is.\n"
    #         )
    #         for reactant, reactions in grouped_reactions.items():
    #
    #             bonds_energy = dict()
    #             for bond, rxns in reactions.items():
    #                 bond = tuple(sorted(reactant.graph_bond_idx_to_ob_bond_idx(bond)))
    #                 bonds_energy[bond] = None
    #                 for charge, attr in rxns.items():
    #                     if attr:
    #                         if bonds_energy[bond] is not None:
    #                             bonds_energy[bond] = min(
    #                                 bonds_energy[bond], attr["bond_energy"]
    #                             )
    #                         else:
    #                             bonds_energy[bond] = attr["bond_energy"]
    #
    #             # write charge
    #             f.write("{}    ".format(reactant.charge))
    #
    #             # write bond energies in the same order as sdf file
    #             sdf_bonds = reactant.get_sdf_bond_indices()
    #             for bond in sdf_bonds:
    #                 energy = bonds_energy[bond]
    #                 if energy is None:
    #                     f.write("0.0 ")
    #                 else:
    #                     f.write("{:.15g} ".format(energy))
    #             f.write("   ")
    #
    #             # write bond energy indicator
    #             for bond in sdf_bonds:
    #                 energy = bonds_energy[bond]
    #                 if energy is None:
    #                     f.write("0 ")
    #                 else:
    #                     f.write("1 ")
    #             f.write("\n")
    #
    #     # write sdf
    #     self.write_sdf(grouped_reactions, struct_file)
    #
    #     # write feature
    #     if feature_file is not None:
    #         self.write_feature(grouped_reactions, feature_file)

    @staticmethod
    def _get_formula_composition_map(mols):
        fcmap = dict()
        for m in mols:
            fcmap[m.formula] = m.composition_dict
        return fcmap

    @staticmethod
    def _is_even_composition(composition):
        for _, amt in composition.items():
            if int(amt) % 2 != 0:
                return False
        return True

    @staticmethod
    def _is_valid_A_to_B_C_composition(composition1, composition2, composition3):
        combined23 = defaultdict(float)
        for k, v in composition2.items():
            combined23[k] += v
        for k, v in composition3.items():
            combined23[k] += v

        sorted1 = sorted(composition1.keys())
        sorted23 = sorted(combined23.keys())
        if sorted1 != sorted23:
            return False
        for k in sorted1:
            if int(composition1[k]) != int(combined23[k]):
                return False
        return True

    @staticmethod
    def _get_molecules_from_reactions(reactions):
        mols = []
        for r in reactions:
            mols.extend(r.reactants + r.products)
        return list(set(mols))

    @staticmethod
    def _is_valid_A_to_B_C_charge(charge1, charge2, charge3):
        return charge1 == charge2 + charge3

    def to_file(self, filename="rxns.pkl"):
        logger.info("Start writing reactions to file: {}".format(filename))

        for m in self.molecules:
            m.make_picklable()
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "molecules": self.molecules,
            "reactions": self.reactions,
        }
        pickle_dump(d, filename)

    @classmethod
    def from_file(cls, filename):
        d = pickle_load(filename)
        logger.info(
            "{} reactions loaded from file: {}".format(len(d["reactions"]), filename)
        )
        return cls(d["molecules"], d["reactions"])

    def to_file_by_ids(self, filename="rxns.yaml"):
        logger.info("Start writing reactions by ids to file: {}".format(filename))
        reaction_ids = []
        for r in self.reactions:
            reaction_ids.append(r.as_dict())
        yaml_dump(reaction_ids, filename)

    @classmethod
    def from_file_by_ids(cls, filename, db_path):
        logger.info("Start loading reactions by ids from file: {}".format(filename))

        db = DatabaseOperation.from_file(db_path)
        mols = db.to_molecules(purify=True)
        id_to_mol_map = {m.id: m for m in mols}

        reactions = yaml_load(filename)

        rxns = []
        for r in tqdm(reactions):
            reactants = [id_to_mol_map[i] for i in r["reactants"]]
            products = [id_to_mol_map[i] for i in r["products"]]
            broken_bond = r["broken_bond"]
            rxn = Reaction(reactants, products, broken_bond)
            rxns.append(rxn)

        logger.info("Finish loading {} reactions".format(len(reactions)))

        return cls(mols, rxns)

    @staticmethod
    def isomorphic_atom_mapping(mol1, mol2):
        """
        Returns `None` is mol1 is not isomorphic to mol2, otherwise the atom mapping from
        mol1 to mol2.
        """
        mol_g1 = mol1.mol_graph
        mol_g2 = mol2.mol_graph
        nx_g1 = mol1.nx_graph
        nx_g2 = mol2.nx_graph
        if len(mol_g1.molecule) != len(mol_g2.molecule):
            return None
        elif (
            mol_g1.molecule.composition.alphabetical_formula
            != mol_g2.molecule.composition.alphabetical_formula
        ):
            return None
        elif len(nx_g1.edges()) != len(nx_g2.edges()):
            return None
        else:
            nm = iso.categorical_node_match("specie", "ERROR")
            GM = iso.GraphMatcher(nx_g1, nx_g2, node_match=nm)
            if GM.is_isomorphic():
                return GM.mapping
            else:
                return None


def is_valid_A_to_B_reaction(reactant, product):
    """
    A -> B
    Args:
        reactant: mol
        product: mol

    Returns:
        A tuple of the bond indices, if this is valid reaction;
        None, otherwise.
    """
    for edge, mgs in reactant.fragments.items():
        if len(mgs) == 1 and mgs[0].isomorphic_to(product.mol_graph):
            return edge
    return None


def is_valid_A_to_B_C_reaction(reactant, products):
    """
    A -> B + C
    Args:
        reactant: mol
        products: list of mols

    Returns:
        A tuple of the bond indices, if this is valid reaction;
        None, otherwise.
    """
    for edge, mgs in reactant.fragments.items():
        if len(mgs) == 2:
            if (
                mgs[0].isomorphic_to(products[0].mol_graph)
                and mgs[1].isomorphic_to(products[1].mol_graph)
            ) or (
                mgs[0].isomorphic_to(products[1].mol_graph)
                and mgs[1].isomorphic_to(products[0].mol_graph)
            ):
                return edge
    return None


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
    fragments1 = reactant1.get_fragments(bonds1)
    fragments2 = reactant2.get_fragments(bonds2)

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
