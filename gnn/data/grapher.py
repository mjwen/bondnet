"""
Build molecule graph and then featurize it.
"""
import dgl


class HomoBidirectedGraph:
    """
    Convert a RDKit molecule to a homogeneous bidirected DGLGraph and featurize for it.

    This creates a bidirectional graph. Atom i of the molecule is node i of the graph.
    Bond 0 corresponds to graph edges 0 and 1, bond 1 corresponds to graph edges 2,
    and 3 ... If `self_loop` is `True`, graph edge 2N represents self loop of atom 0,
    edge 2N+1 represents self loop of atom 1... where N is the number of bonds in the
    molecule.

    Notes:
        Make sure your featurizer match the above order, and pay carefully attention
        to bond featurizer.
    """

    def __init__(self, atom_featurizer=None, bond_featurizer=None, self_loop=True):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.self_loop = self_loop

    def build_graph_and_featurize(self, mol, **kwargs):
        """
        Build a graph with atoms as the nodes and bonds as the edges and then featurize
        the graph.

        Args:
            mol (rdkit mol): a rdkit molecule
            kwargs: extra keyword arguments needed by featurizer

        Returns:
            (DGLGraph)
        """

        g = self.build_graph(mol)
        g = self.featurize(g, mol, **kwargs)
        return g

    def build_graph(self, mol):
        g = dgl.DGLGraph()

        # Add nodes
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        # Add edges
        src_list = []
        dst_list = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
        g.add_edges(src_list, dst_list)

        if self.self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        return g

    def featurize(self, g, mol, **kwargs):
        if self.atom_featurizer is not None:
            g.ndata.update(self.atom_featurizer(mol, **kwargs))
        if self.bond_featurizer is not None:
            g.edata.update(self.bond_featurizer(mol, **kwargs))
        return g


class HomoCompleteGraph:
    """
    Convert a RDKit molecule to a homogeneous complete DGLGraph and featurize for it.

    This creates a complete graph, i.e. every atom is connected to other atoms in the
    molecule. If `self_loop` is `True`, each atom is connected to its self.

    The edges are in the order of (0, 0), (0, 1), (0, 2), ... (1, 0), (1, 1), (1, 2),
     ... If not `self_loop` are not created, we will not have (0, 0), (1, 1), ...
    """

    def __init__(self, atom_featurizer=None, bond_featurizer=None, self_loop=True):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.self_loop = self_loop

    def build_graph_and_featurize(self, mol, **kwargs):
        """
        Build a graph with atoms as the nodes and bonds as the edges and then featurize
        the graph.


        Args:
            mol (rdkit mol): a rdkit molecule
            kwargs: extra keyword arguments needed by featurizer

        Returns:
            (DGLGraph)
        """

        g = self.build_graph(mol)
        g = self.featurize(g, mol, **kwargs)
        return g

    def build_graph(self, mol):

        g = dgl.DGLGraph()
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        if self.self_loop:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms)],
                [j for i in range(num_atoms) for j in range(num_atoms)],
            )
        else:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms - 1)],
                [j for i in range(num_atoms) for j in range(num_atoms) if i != j],
            )

        return g

    def featurize(self, g, mol, **kwargs):
        if self.atom_featurizer is not None:
            g.ndata.update(self.atom_featurizer(mol, **kwargs))
        if self.bond_featurizer is not None:
            g.edata.update(self.bond_featurizer(mol, **kwargs))
        return g


class HeteroMoleculeGraph:
    """
    Convert a RDKit molecule to a DGLHeteroGraph and featurize for it.

    Atom, bonds, and global states are all represented as nodes in the graph.
    Atom i corresponds to graph node (of type `atom`) i.
    Bond i corresponds to graph node (of type `bond`) i.
    There is only one global state node 0.
    """

    def __init__(
        self,
        atom_featurizer=None,
        bond_featurizer=None,
        global_state_featurizer=None,
        self_loop=True,
    ):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_state_featurizer = global_state_featurizer
        self.self_loop = self_loop

    def build_graph_and_featurize(self, mol, **kwargs):
        """
        Build an a heterograph, with three types of nodes: atom, bond, and glboal
        state, and then featurize the graph.

        Args:
            mol (rdkit mol): a rdkit molecule
            kwargs: extra keyword arguments needed by featurizer

        Returns:
            (dgl heterograph): bond_idx_to_atom_idx (dict): mapping between two type
                bond indices, key is integer bond index, and value is a tuple of atom
                indices that specify the bond.
        """

        g = self.build_graph(mol)
        g = self.featurize(g, mol, **kwargs)
        return g

    def build_graph(self, mol):
        num_atoms = mol.GetNumAtoms()

        # bonds
        num_bonds = mol.GetNumBonds()
        bond_idx_to_atom_idx = dict()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_idx_to_atom_idx[i] = (u, v)

        a2b = []
        b2a = []
        for a in range(num_atoms):
            for b, bond in bond_idx_to_atom_idx.items():
                if a in bond:
                    b2a.append([b, a])
                    a2b.append([a, b])

        a2g = [(a, 0) for a in range(num_atoms)]
        g2a = [(0, a) for a in range(num_atoms)]
        b2g = [(b, 0) for b in range(num_bonds)]
        g2b = [(0, b) for b in range(num_bonds)]

        edges_dict = {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): a2g,
            ("global", "g2a", "atom"): g2a,
            ("bond", "b2g", "global"): b2g,
            ("global", "g2b", "bond"): g2b,
        }
        if self.self_loop:
            a2a = [(i, i) for i in range(num_atoms)]
            b2b = [(i, i) for i in range(num_bonds)]
            g2g = [(0, 0)]
            edges_dict.update(
                {
                    ("atom", "a2a", "atom"): a2a,
                    ("bond", "b2b", "bond"): b2b,
                    ("global", "g2g", "global"): g2g,
                }
            )
        g = dgl.heterograph(edges_dict)

        return g

    def featurize(self, g, mol, **kwargs):

        if self.atom_featurizer is not None:
            g.nodes["atom"].data.update(self.atom_featurizer(mol, **kwargs))
        if self.bond_featurizer is not None:
            g.nodes["bond"].data.update(self.bond_featurizer(mol, **kwargs))
        if self.global_state_featurizer is not None:
            g.nodes["global"].data.update(self.global_state_featurizer(mol, **kwargs))

        return g
