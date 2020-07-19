"""
Test the Heterograph conv.
"""
from ..utils import make_hetero_CH2O
from bondnet.model.hgat_bond import HGATBond


def test_hgat_bond():

    g, feats = make_hetero_CH2O()

    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a"], "nodes": ["bond", "global"]},
        "bond": {"edges": ["a2b", "g2b"], "nodes": ["atom", "global"]},
        "global": {"edges": ["a2g", "b2g"], "nodes": ["atom", "bond"]},
    }
    attn_order = ["atom", "bond", "global"]
    in_feats = [feats[t].shape[1] for t in attn_order]

    nbonds = 3
    nmols = 1

    model = HGATBond(attn_mechanism, attn_order, in_feats)
    output = model(g, feats)
    assert tuple(output.shape) == (nbonds,)

    model = HGATBond(attn_mechanism, attn_order, in_feats)
    output = model(g, feats, mol_based=True)
    assert tuple(output.shape) == (nmols, 1)

    outdim = 2
    model = HGATBond(
        attn_mechanism, attn_order, in_feats, outdim=outdim, classification=True
    )
    output = model(g, feats)
    assert tuple(output.shape) == (nbonds, outdim)

    outdim = 2
    model = HGATBond(
        attn_mechanism, attn_order, in_feats, outdim=outdim, classification=True
    )
    output = model(g, feats, mol_based=True)
    assert tuple(output[0].shape) == (nbonds, outdim)
