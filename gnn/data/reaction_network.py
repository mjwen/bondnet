import random
import numpy as np
import itertools
import dgl


class Reaction:
    def __init__(
        self, reactants, products, atom_mapping=None, bond_mapping=None, id=None
    ):
        """
        A class to represent chemical reaction.

        Args:
            reactants (list): integer indices of reactants
            products (list): integer indices of reactants
            atom_mapping (list of dict): each dict is an atom mapping from product to
                reactant
            bond_mapping (list of dict): each dict is a bond mapping from product to
                reactant
            id (int or str): unique identifier of the reaction
        """
        self._reactants = reactants
        self._products = products
        self.atom_mapping = atom_mapping
        self.bond_mapping = bond_mapping
        self.id = id

        self._atom_mapping_list = None
        self._bond_mapping_list = None

    @property
    def reactants(self):
        return self._reactants

    @reactants.setter
    def reactants(self, reactants):
        self._reactants = reactants

    @property
    def products(self):
        return self._products

    @products.setter
    def products(self, products):
        self._products = products

    @property
    def atom_mapping_as_list(self):
        if self._atom_mapping_list is None:
            assert (
                self.atom_mapping is not None
            ), "atom_mapping not provided at instantiation"
            self._atom_mapping_list = self._mapping_as_list(self.atom_mapping)
        return self._atom_mapping_list

    @property
    def bond_mapping_as_list(self):
        if self._bond_mapping_list is None:
            assert (
                self.bond_mapping is not None
            ), "bond_mapping not provided at instantiation"
            self._bond_mapping_list = self._mapping_as_list(self.bond_mapping)
        return self._bond_mapping_list

    @staticmethod
    def _mapping_as_list(mappings):
        """
        Given a list of mappings from products to reactants, return a representation
        (mapping expressed using a list) from reactants to products.
        Args:
            mappings (list of dict): mappings from products to reactants

        Returns:
            list: mapping from reactant to product

        Example:
            >>>mappings = [{0:1, 1:3}, {0:2, 1:0}]
            >>>_mapping_as_list(mappings)
            >>>[3,0,2,1]
        """

        sizes = [len(mp) for mp in mappings]
        accumulate = [i for i in itertools.accumulate(sizes)]
        accumulate = [0] + accumulate[:-1]

        # combined mapping from products to reactants
        # items in the first mapping is simply copied
        # item 0 in the second mapping has a key = len(mappings[0]) + key
        # ...
        combined_mapping = {}
        for i, mp in enumerate(mappings):
            for p, r in mp.items():
                combined_mapping[p + accumulate[i]] = r

        # mapping from reactants to products
        r2p_mapping = {v: k for k, v in combined_mapping.items()}

        # represent the mapping as a list, where the reactant item is indexed by the
        # list index
        mp_list = [r2p_mapping[k] for k in sorted(r2p_mapping)]

        return mp_list


class ReactionNetwork:
    """
    Args:
        molecules (list): a sequence of Molecules
        reactions (list): a sequence of Reaction.
    """

    def __init__(self, molecules, reactions):
        self.molecules = np.asarray(molecules)
        self.reactions = np.asarray(reactions)

        m2r = []
        r2m = []
        for i, rxn in enumerate(reactions):
            m2r += [(m, i) for m in rxn.reactants]
            r2m += [(i, m) for m in rxn.products]

        edges_dict = {
            ("molecule", "m2r", "reaction"): m2r,
            ("reaction", "r2m", "molecule"): r2m,
        }

        # create graph
        self.g = dgl.heterograph(edges_dict)

        # attach molecules to graph
        self.g.nodes["molecule"].data.update({"mol": self.molecules})

    @staticmethod
    def _get_mol_ids_from_reactions(reactions):
        """
        Get the ids of all molecules participating the reactions.

        Args:
            reactions (list): a sequence of `Reaction`.

        Returns:
            list: molecules (integer ids)

        """
        mol_ids = set()
        for rxn in reactions:
            mol_ids.update(rxn.reactants + rxn.products)
        return sorted(mol_ids)

    def subselect_reactions(self, indices=None):
        """
        Subselect some reactions in the network and get all molecules in the
        subset of reactions.

        Args:
            indices (int or list): If `int`, randomly select a subset of `indices`
                reactions from the network. If `list`, select the reactions specified
                by `indices`.

        Returns:
            sub_reactions (list): a sequence of `Reaction`. The indices of reactants and
                products of each reaction are remapped from global index to the
                index in the subset of molecules.
            sub_molecules (list): all molecules in the selected subset of reactions.
        """
        if isinstance(indices, int):
            x = list(range(len(self.reactions)))
            random.shuffle(x)
            indices = x[:indices]

        # reactions subset
        sub_reactions = self.reactions[indices]

        # subset ids and map between global molecule index and subset molecule index
        ids = self._get_mol_ids_from_reactions(sub_reactions)
        global_to_subset_mapping = {g: s for s, g in enumerate(ids)}

        # change global molecule index to subset molecule index in reaction
        for r in sub_reactions:
            r.reactants = [global_to_subset_mapping[i] for i in r.reactants]
            r.products = [global_to_subset_mapping[i] for i in r.products]

        # molecules subset
        sub_molecules = self.molecules[ids]

        return sub_reactions, sub_molecules
