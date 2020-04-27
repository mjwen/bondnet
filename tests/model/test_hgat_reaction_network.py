import numpy as np
from gnn.data.reaction_network import Reaction
from gnn.model.hgat_reaction_network import (
    create_rxn_graph,
    mol_graph_to_rxn_graph,
    HGATReactionNetwork,
)

from ..utils import (
    make_hetero_CH2O,
    make_hetero_CHO,
    make_hetero_H,
    make_batched_hetero_forming_reaction,
)


def test_create_rxn_graph():

    # create reactant and products
    reactant, _ = make_hetero_CH2O()
    products = []
    g, _ = make_hetero_CHO()
    products.append(g)
    g, _ = make_hetero_H()
    products.append(g)

    mappings = {
        "atom": [0, 1, 2, 3],
        "bond": [0, 1, 2],
    }
    has_bonds = {"reactants": [True], "products": [True, False]}
    graph, feats = create_rxn_graph(
        [reactant], products, mappings, has_bonds, ft_name="feat"
    )

    ref_feats = {
        "atom": [[0, 0], [0, 0], [0, 0], [-6, -6]],
        "bond": [[0, 0, 0], [0, 0, 0], [-6, -7, -8]],
        "global": [[0, 1, 2, 3]],
    }

    for nt in graph.ntypes:
        assert np.allclose(feats[nt], ref_feats[nt])


def test_mol_graph_to_rxn_graph():
    graph, feats = make_batched_hetero_forming_reaction()

    rxn = Reaction(
        reactants=[0],
        products=[1, 2],
        atom_mapping=[{0: 0, 1: 1, 2: 2}, {0: 3}],
        bond_mapping=[{0: 0, 1: 1}, {}],
    )

    reactions = [rxn, rxn]

    graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)

    ref_feats = {
        "atom": [[0, 0], [0, 0], [0, 0], [-6, -6], [0, 0], [0, 0], [0, 0], [-6, -6]],
        "bond": [[0, 0, 0], [0, 0, 0], [-6, -7, -8], [0, 0, 0], [0, 0, 0], [-6, -7, -8]],
        "global": [[0, 1, 2, 3], [0, 1, 2, 3]],
    }

    for nt in graph.ntypes:
        x = nt
        assert np.allclose(feats[nt], ref_feats[nt])


def test_hgat_reaction_network():

    g, feats = make_batched_hetero_forming_reaction()

    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a"], "nodes": ["bond", "global"]},
        "bond": {"edges": ["a2b", "g2b"], "nodes": ["atom", "global"]},
        "global": {"edges": ["a2g", "b2g"], "nodes": ["atom", "bond"]},
    }
    attn_order = ["atom", "bond", "global"]
    in_feats = [feats[t].shape[1] for t in attn_order]

    model = HGATReactionNetwork(attn_mechanism, attn_order, in_feats, outdim=3)

    rxn = Reaction(
        reactants=[0],
        products=[1, 2],
        atom_mapping=[{0: 0, 1: 1, 2: 2}, {0: 3}],
        bond_mapping=[{0: 0, 1: 1}, {}],
    )
    reactions = [rxn, rxn]

    output = model(g, feats, reactions)

    assert tuple(output.shape) == (2, 3)
