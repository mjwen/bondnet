"""
Test the Heterograph conv.
"""
from ..utils import make_hetero_CH2O
from gnn.model.hgat_bond import HGATBond


def test_hgat_bond():

    g, feats = make_hetero_CH2O()

    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a"], "nodes": ["bond", "global"]},
        "bond": {"edges": ["a2b", "g2b"], "nodes": ["atom", "global"]},
        "global": {"edges": ["a2g", "b2g"], "nodes": ["atom", "bond"]},
    }
    attn_order = ["atom", "bond", "global"]
    in_feats = [feats[t].shape[1] for t in attn_order]

    model = HGATBond(attn_mechanism, attn_order, in_feats)
    output = model(g, feats)
    assert tuple(output.shape) == (3,)

    model = HGATBond(attn_mechanism, attn_order, in_feats)
    output = model(g, feats, mol_energy=True)
    assert tuple(output.shape) == (1, 1)

    model = HGATBond(attn_mechanism, attn_order, in_feats, outdim=3, classification=True)
    output = model(g, feats)
    assert tuple(output[0].shape) == (3, 3)
