import itertools
import logging
import json
import networkx as nx
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from gnn.data.database import DatabaseOperation
from gnn.utils import create_directory, pickle_dump, pickle_load, yaml_dump, expand_path

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

    def as_dict(self):
        d = {
            "reactants": ["{}_{}".format(m.formula, m.id) for m in self.reactants],
            "products": ["{}_{}".format(m.formula, m.id) for m in self.products],
            "charge": [m.charge for m in self.reactants + self.products],
            "broken_bond": self.broken_bond,
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


class ReactionExtractor:
    def __init__(self, molecules, reactions=None):
        self.molecules = molecules or self._get_molecules_from_reactions(reactions)
        self.reactions = reactions

        self.buckets = None
        self.bucket_keys = None

    def get_molecule_properties(self, keys):
        values = defaultdict(list)
        for m in self.molecules:
            for k in keys:
                values[k].append(getattr(m, k))
        return values

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
                print("@@flag running bucket", i)
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
                print("@@flag running bucket", i)

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
                        ids = set([A.id, B.id, C.id])
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

    def group_by_reactant(self, string_reactant_index=False):
        """
        Group reactions to dict with reactant as the key and list of reactions as the
        value.
        """
        grouped_reactions = defaultdict(list)
        for rxn in self.reactions:
            reactant = rxn.reactants[0]
            reactant_idx = self._get_reactant_index(reactant, string_reactant_index)
            grouped_reactions[reactant_idx].append(rxn)
        return grouped_reactions

    def group_by_reactant_charge_and_bond(
        self, string_reactant_index=False, babel_bond_indices=True
    ):
        """
        Group all the reactions according to the reactant and the charge of the
        reactant and products.

        Args:
            string_reactant_index: If True, the reactant id (a string) is used as the
                key for the outer dict, otherwise, the reactant instance is used.
            babel_bond_indices: If True, babel atoms (indices) will be used as the key
                to denote broken bond for the inner dict, otherwise, mol_graph nodes
                will be used.

        Returns:
            A dict of dict of dict. The outer dict has reactant index as the key,
            the middle dict has charges (a tuple) as the key and the inner dict has bond
            indices (a tuple) as the key and bond attributes (a dict of energy,
            bond order, ect.).
        """
        grouped_reactions = self.group_by_reactant(string_reactant_index=False)

        groups = OrderedDict()
        for reactant, reactions in grouped_reactions.items():
            reactant_idx = self._get_reactant_index(reactant, string_reactant_index)
            groups[reactant_idx] = dict()

            for rxn in reactions:
                charge = tuple([m.charge for m in (rxn.reactants + rxn.products)])
                groups[reactant_idx][charge] = OrderedDict()
                for bond in reactant.graph.edges():
                    bond_indices = self._get_bond_indices(
                        bond, reactant, babel_bond_indices
                    )
                    groups[reactant_idx][charge][bond_indices] = None

            for rxn in reactions:
                charge = tuple([m.charge for m in (rxn.reactants + rxn.products)])
                bond_indices = self._get_bond_indices(
                    rxn.get_broken_bond(), reactant, babel_bond_indices
                )
                groups[reactant_idx][charge][bond_indices] = rxn.as_dict()

        return groups

    def group_by_reactant_bond_and_charge(
        self, string_reactant_index=False, babel_bond_indices=True
    ):
        """
        Group all the reactions to nested dicts according to the reactant, bond, and
        charge.

        Args:
            string_reactant_index: If True, the reactant id (a string) is used as the
                key for the outer dict, otherwise, the reactant instance is used.
            babel_bond_indices: If True, babel atoms (indices) will be used as the key
                to denote broken bond for the inner dict, otherwise, mol_graph nodes
                will be used.

        Returns:
            A dict of dict of dict. The outer dict has reactant index as the key,
            the middle dict has charges (a tuple) as the key and the inner dict has bond
            indices (a tuple) as the key and bond attributes (a dict of energy,
            bond order, ect.).
        """
        grouped_reactions = self.group_by_reactant(string_reactant_index=False)

        groups = OrderedDict()
        for reactant, reactions in grouped_reactions.items():
            reactant_idx = self._get_reactant_index(reactant, string_reactant_index)
            groups[reactant_idx] = OrderedDict()

            for bond in reactant.graph.edges():
                bond_indices = self._get_bond_indices(bond, reactant, babel_bond_indices)
                groups[reactant_idx][bond_indices] = dict()

            for rxn in reactions:
                charge = tuple([m.charge for m in (rxn.reactants + rxn.products)])
                bond_indices = self._get_bond_indices(
                    rxn.get_broken_bond(), reactant, babel_bond_indices
                )
                groups[reactant_idx][bond_indices][charge] = rxn.as_dict()

        return groups

    def bond_energies_to_file(
        self,
        filename,
        mode="reactant_bond_and_charge",
        string_reactant_index=True,
        babel_bond_indices=True,
    ):
        if mode == "reactant_bond_and_charge":
            groups = self.group_by_reactant_bond_and_charge(
                string_reactant_index, babel_bond_indices
            )
        elif mode == "reactant_charge_and_bond":
            groups = self.group_by_reactant_charge_and_bond(
                string_reactant_index, babel_bond_indices
            )
        else:
            raise RuntimeError("mode unsupported")
        for m in self.molecules:
            m.make_picklable()
        yaml_dump(groups, filename)

    def write_sdf(self, molecules, filename="molecules.sdf"):
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
                sdf = m.write(file_format="sdf", mol_id=m.id + " int_id-" + str(i))
                f.write(sdf)

    def create_struct_label_dataset_with_lowest_energy_across_charge(
        self, struct_name="sturct.sdf", label_name="label.txt"
    ):
        """
        Write the reactions to files.

        Each reactant may have multiple products corresponding to various charge
        combinations. Here, we write the lowest energy one.
        """
        grouped_reactions = self.group_by_reactant_bond_and_charge(False, True)

        # write sdf
        self.write_sdf(grouped_reactions, struct_name)

        # write label
        label_name = expand_path(label_name)
        create_directory(label_name)
        with open(label_name, "w") as f:
            f.write(
                "# Each line lists the molecule charge and bond energies of a molecule. "
                "The number of items in each line is equal to 1 + 2*N, where N is the "
                "number bonds. The first item is the molecule charge. The first half "
                "of the remaining items are bond energies and the next half values are "
                "indicators (0 or 1) to specify whether the bond energy exist in the "
                "dataset. A value of 0 means the corresponding bond energy should be "
                "ignored, whatever its value is.\n"
            )
            for reactant, reactions in grouped_reactions.items():

                bonds_energy = dict()
                for bond, rxns in reactions.items():
                    bonds_energy[bond] = None
                    for _, attr in rxns.items():
                        if attr:
                            if bonds_energy[bond] is not None:
                                bonds_energy[bond] = min(
                                    bonds_energy[bond], attr["bond_energy"]
                                )
                            else:
                                bonds_energy[bond] = attr["bond_energy"]

                # write charge
                f.write("{}    ".format(reactant.charge))

                # write bond energies in the same order as sdf file
                sdf_bonds = reactant.get_sdf_bond_indices()
                for bond in sdf_bonds:
                    energy = bonds_energy[bond]
                    if energy is None:
                        f.write("0.0 ")
                    else:
                        f.write("{:.15g} ".format(energy))
                f.write("   ")

                # write bond energy indicator
                for bond in sdf_bonds:
                    energy = bonds_energy[bond]
                    if energy is None:
                        f.write("0 ")
                    else:
                        f.write("1 ")
                f.write("\n")

    def create_struct_label_dataset(
        self, struct_name="sturct.sdf", label_name="label.txt"
    ):
        """
        Write the reactions to files.

        Each reactant may have multiple products corresponding to various charge
        combinations. Here, we write all of them.
        """
        grouped_reactions = self.group_by_reactant_charge_and_bond(False, True)

        molecules = []

        label_name = expand_path(label_name)
        create_directory(label_name)
        with open(label_name, "w") as f:
            f.write(
                "# Each line lists the bond energies of a molecule. "
                "It contains four parts: "
                "1) the number of substances N (i.e. number of reactants and products) "
                "2) N numbers of the charge of the substances "
                "3) the remaining part is equal to 2 times the number of bonds, "
                "where the first half values are bond energies and "
                "4) the next half values are indicators "
                " (0 or 1) to specify whether the bond energy exist in the dataset. a "
                "value of 0 means the corresponding bond energy will be ignored, "
                "no matter what its value is.\n"
            )
            for reactant, reactions in grouped_reactions.items():
                sdf_bonds = reactant.get_sdf_bond_indices()

                for charge, rxns in reactions.items():
                    molecules.append(reactant)

                    # number of substance and substance charges
                    f.write("{} ".format(len(charge)))
                    for c in charge:
                        f.write("{} ".format(c))
                    f.write("   ")

                    bonds_energy = dict()
                    for bond, attr in rxns.items():
                        if attr is not None:
                            bonds_energy[bond] = attr["bond_energy"]
                        else:
                            bonds_energy[bond] = None

                    # write bond energies in the same order as sdf file
                    for bond in sdf_bonds:
                        energy = bonds_energy[bond]
                        if energy is None:
                            f.write("0.0 ")
                        else:
                            f.write("{:.15g} ".format(energy))
                    f.write("   ")

                    # write bond energy indicator
                    for bond in sdf_bonds:
                        energy = bonds_energy[bond]
                        if energy is None:
                            f.write("0 ")
                        else:
                            f.write("1 ")
                    f.write("\n")

        # write sdf
        self.write_sdf(molecules, struct_name)

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
    def _get_bond_indices(bond, reactant, use_babel_bond_indices):
        """
        Convert mol_graph bond indices to babel bond indices.
        """
        if use_babel_bond_indices:
            idx0 = reactant.graph_idx_to_ob_idx_map[bond[0]]
            idx1 = reactant.graph_idx_to_ob_idx_map[bond[1]]
            bond_indices = tuple(sorted([idx0, idx1]))
        else:
            bond_indices = bond
        return bond_indices

    @staticmethod
    def _get_reactant_index(reactant, use_string_reactant_index):
        """
        Convert reactant to an identifier, which is used as index.
        """
        if use_string_reactant_index:
            reactant_idx = reactant.formula + "_" + reactant.id
        else:
            reactant_idx = reactant
        return reactant_idx

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

    def to_file_by_ids(self, filename="rxns.json"):
        logger.info("Start writing reactions by ids to file: {}".format(filename))
        reaction_ids = []
        for r in self.reactions:
            reaction_ids.append(r.as_dict())
        with open(filename, "w") as f:
            json.dump(reaction_ids, f)

    @classmethod
    def from_file_by_ids(cls, filename, db_path):
        logger.info("Start loading reactions by ids from file: {}".format(filename))

        db = DatabaseOperation.from_file(db_path)
        mols = db.to_molecules(purify=True)
        id_to_mol_map = {m.id: m for m in mols}

        with open(filename, "r") as f:
            reactions = json.load(f)

        rxns = []
        for r in tqdm(reactions):
            reactants = [id_to_mol_map[i] for i in r["reactants"]]
            products = [id_to_mol_map[i] for i in r["products"]]
            broken_bond = r["broken_bond"]
            rxn = Reaction(reactants, products, broken_bond)
            rxns.append(rxn)

        logger.info("Finish loading {} reactions".format(len(reactions)))

        return cls(mols, rxns)


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
