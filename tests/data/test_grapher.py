import numpy as np
from collections import defaultdict
import itertools
from gnn.data.grapher import (
    HomoBidirectedGraph,
    HomoCompleteGraph,
    HeteroMoleculeGraph,
    HeteroCompleteGraph,
)
from gnn.data.utils import get_atom_to_bond_map, get_bond_to_atom_map
from .utils import make_a_mol


def get_hetero_self_loop_map(g, ntype):
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


def test_build_homo_bidirected_graph():
    def assert_graph(self_loop):
        m = make_a_mol()
        natoms = m.GetNumAtoms()
        nbonds = m.GetNumBonds()
        if self_loop:
            nedges = 2 * nbonds + natoms
        else:
            nedges = 2 * nbonds

        grapher = HomoBidirectedGraph(self_loop=self_loop)
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


def test_build_homo_complete_graph():
    def assert_graph(self_loop):
        m = make_a_mol()
        natoms = m.GetNumAtoms()
        if self_loop:
            nedges = natoms ** 2
            edges = zip(
                [i for i in range(natoms) for j in range(natoms)],
                [j for i in range(natoms) for j in range(natoms)],
            )
            edges = [[int(i), int(j)] for i, j in edges]

        else:
            nedges = natoms * (natoms - 1)
            edges = zip(
                [i for i in range(natoms) for j in range(natoms - 1)],
                [j for i in range(natoms) for j in range(natoms) if i != j],
            )
            edges = [[int(i), int(j)] for i, j in edges]

        grapher = HomoCompleteGraph(self_loop=self_loop)
        g = grapher.build_graph(m)

        assert g.number_of_nodes() == natoms
        assert g.number_of_edges() == nedges

        graph_edges = np.array([[int(i), int(j)] for i, j in zip(*g.edges())])

        assert np.array_equal(edges, graph_edges)

    assert_graph(False)
    assert_graph(True)


def test_build_hetero_graph():
    def assert_graph(self_loop):
        m = make_a_mol()
        grapher = HeteroMoleculeGraph(self_loop=self_loop)
        g = grapher.build_graph(m)

        # number of atoms
        na = 7
        # number of bonds
        nb = 7
        # number of edges between atoms and bonds
        ne = 2 * nb

        nodes = ["atom", "bond", "global"]
        num_nodes = [g.number_of_nodes(n) for n in nodes]
        ref_num_nodes = [na, 7, 1]

        if self_loop:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb, na, nb, 1]

        else:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb]

        assert set(g.ntypes) == set(nodes)
        assert set(g.etypes) == set(edges)
        assert num_nodes == ref_num_nodes
        assert num_edges == ref_num_edges

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

        if self_loop:
            for nt, n in zip(nodes, num_nodes):
                assert get_hetero_self_loop_map(g, nt) == {i: [i] for i in range(n)}

    assert_graph(False)
    assert_graph(True)


def test_build_complete_hetero_graph():
    def assert_graph(self_loop):
        m = make_a_mol()
        grapher = HeteroCompleteGraph(self_loop=self_loop)
        g = grapher.build_graph(m)

        # number of atoms
        na = 7
        # number of bonds
        nb = na * (na - 1) // 2
        # number of edges between atoms and bonds
        ne = 2 * nb

        nodes = ["atom", "bond", "global"]
        num_nodes = [g.number_of_nodes(n) for n in nodes]
        ref_num_nodes = [na, na * (na - 1) // 2, 1]

        if self_loop:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb, na, nb, 1]

        else:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb]

        assert set(g.ntypes) == set(nodes)
        assert set(g.etypes) == set(edges)
        assert num_nodes == ref_num_nodes
        assert num_edges == ref_num_edges

        bond_to_atom_map = {}
        atom_to_bond_map = defaultdict(list)
        i = 0
        for u in range(na):
            for v in range(u + 1, na):
                bond_to_atom_map[i] = sorted([u, v])
                atom_to_bond_map[u].append(i)
                atom_to_bond_map[v].append(i)
                i += 1
        atom_to_bond_map = {a: sorted(bonds) for a, bonds in atom_to_bond_map.items()}

        ref_b2a_map = get_bond_to_atom_map(g)
        ref_a2b_map = get_atom_to_bond_map(g)
        assert bond_to_atom_map == ref_b2a_map
        assert atom_to_bond_map == ref_a2b_map

        if self_loop:
            for nt, n in zip(nodes, num_nodes):
                assert get_hetero_self_loop_map(g, nt) == {i: [i] for i in range(n)}

    assert_graph(False)
    assert_graph(True)
