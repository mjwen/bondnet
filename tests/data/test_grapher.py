import numpy as np
from collections import defaultdict
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeFeaturizer,
    MolChargeFeaturizer,
)
from gnn.data.grapher import HomoMoleculeGraph, HeteroMoleculeGraph
from gnn.data.utils import get_atom_to_bond_map, get_bond_to_atom_map
from .utils import make_EC_mol


def test_build_hetero_graph():
    m = make_EC_mol()
    grapher = HeteroMoleculeGraph(self_loop=False)
    g = grapher.build_graph(m)

    nodes = ["atom", "bond", "global"]
    edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
    assert set(g.ntypes) == set(nodes)
    assert set(g.etypes) == set(edges)
    num_nodes = [g.number_of_nodes(n) for n in nodes]
    assert num_nodes == [7, 7, 1]
    num_edges = [g.number_of_edges(e) for e in edges]
    assert num_edges == [14, 14, 7, 7, 7, 7]

    bond_to_atom_map = {
        0: [0, 1],
        1: [0, 4],
        2: [1, 2],
        3: [1, 6],
        4: [2, 3],
        5: [3, 4],
        6: [0, 5],
    }
    atom_to_bond_map = defaultdict(list)
    for b, atoms in bond_to_atom_map.items():
        for a in atoms:
            atom_to_bond_map[a].append(b)
    atom_to_bond_map = {a: sorted(bonds) for a, bonds in atom_to_bond_map.items()}

    ref_b2a_map = get_bond_to_atom_map(g)
    ref_a2b_map = get_atom_to_bond_map(g)
    assert bond_to_atom_map == ref_b2a_map
    assert atom_to_bond_map == ref_a2b_map


def test_build_hetero_graph_self_loop():
    m = make_EC_mol()
    grapher = HeteroMoleculeGraph(self_loop=True)
    g = grapher.build_graph(m)

    nodes = ["atom", "bond", "global"]
    edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
    assert set(g.ntypes) == set(nodes)
    assert set(g.etypes) == set(edges)
    num_nodes = [g.number_of_nodes(n) for n in nodes]
    assert num_nodes == [7, 7, 1]
    num_edges = [g.number_of_edges(e) for e in edges]
    assert num_edges == [14, 14, 7, 7, 7, 7, 7, 7, 1]

    bond_to_atom_map = {
        0: [0, 1],
        1: [0, 4],
        2: [1, 2],
        3: [1, 6],
        4: [2, 3],
        5: [3, 4],
        6: [0, 5],
    }
    atom_to_bond_map = defaultdict(list)
    for b, atoms in bond_to_atom_map.items():
        for a in atoms:
            atom_to_bond_map[a].append(b)
    atom_to_bond_map = {a: sorted(bonds) for a, bonds in atom_to_bond_map.items()}

    ref_b2a_map = get_bond_to_atom_map(g)
    ref_a2b_map = get_atom_to_bond_map(g)
    assert bond_to_atom_map == ref_b2a_map
    assert atom_to_bond_map == ref_a2b_map

    def get_self_loop_map(g, ntype):
        num = g.number_of_nodes(ntype)
        if ntype == "atom":
            etype = "a2a"
        elif ntype == "bond":
            etype = "b2b"
        elif ntype == "global":
            etype = "g2g"
        else:
            raise ValueError("not supported node type: {}".format(ntype))
        self_loop_map = dict()
        for i in range(num):
            suc = g.successors(i, etype)
            self_loop_map[i] = list(suc)
        return self_loop_map

    for nt, n in zip(nodes, num_nodes):
        assert get_self_loop_map(g, nt) == {i: [i] for i in range(n)}


# #  # NOTE similar test s in test_featurizer
# def test_hetero_graph_featurize():
#     m = make_EC_mol()
#     species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
#     charge = 1
#
#     atom_featurizer = AtomFeaturizer(species)
#     bond_featurizer = BondAsNodeFeaturizer()
#     global_state_featurizer = MolChargeFeaturizer()
#     grapher = HeteroMoleculeGraph(
#         atom_featurizer, bond_featurizer, global_state_featurizer
#     )
#     g = grapher.build_graph_and_featurize(m, charge=charge)
#     assert np.array_equal(
#         g.nodes["atom"].data["feat"].shape,
#         (m.GetNumAtoms(), atom_featurizer.feature_size),
#     )
#     assert np.array_equal(
#         g.nodes["bond"].data["feat"].shape,
#         (m.GetNumBonds(), bond_featurizer.feature_size),
#     )
#     assert np.array_equal(
#         g.nodes["global"].data["feat"].shape, (1, global_state_featurizer.feature_size)
#     )


def test_build_homo_graph():
    def assert_graph(self_loop):
        m = make_EC_mol()
        natoms = m.GetNumAtoms()
        nbonds = m.GetNumBonds()
        if self_loop:
            nedges = 2 * nbonds + natoms
        else:
            nedges = 2 * nbonds

        grapher = HomoMoleculeGraph(self_loop=self_loop)
        g = grapher.build_graph(m)

        assert g.number_of_nodes() == natoms
        assert g.number_of_edges() == nedges

        # order matters, connectivity is how we add edges
        c1 = [[1, 2], [1, 5], [2, 3], [2, 7], [4, 3], [5, 4], [6, 1]]
        c2 = [list(reversed(i)) for i in c1]
        connectivity = []
        for i, j in zip(c1, c2):
            connectivity.append(i)
            connectivity.append(j)
        if self_loop:
            connectivity += [[i, i] for i in range(1, natoms + 1)]
        connectivity = np.array(connectivity) - 1  # -1 to convert index to start from 0

        edges = np.array([[int(i), int(j)] for i, j in zip(*g.edges())])

        assert np.array_equal(connectivity, edges)

    assert_graph(False)
    assert_graph(True)


#  # NOTE similar test s in test_featurizer
# def test_homo_graph_featurize():
#     def assert_graph(self_loop):
#         m = make_EC_mol()
#         natoms = m.GetNumAtoms()
#         nbonds = m.GetNumBonds()
#         if self_loop:
#             nedges = 2 * nbonds + natoms
#         else:
#             nedges = 2 * nbonds
#         species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
#
#         atom_featurizer = AtomFeaturizer(species)
#         bond_featurizer = BondAsEdgeFeaturizer(self_loop=self_loop)
#         grapher = HomoMoleculeGraph(atom_featurizer, bond_featurizer, self_loop=self_loop)
#         g = grapher.build_graph_and_featurize(m)
#
#         assert np.array_equal(
#             g.ndata["feat"].shape, (natoms, atom_featurizer.feature_size)
#         )
#         assert np.array_equal(
#             g.edata["feat"].shape, (nedges, bond_featurizer.feature_size)
#         )
#
#     assert_graph(True)
#     assert_graph(False)
