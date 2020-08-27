import torch
import numpy as np
from bondnet.layer.gatedconv import GatedGCNConv, select_not_equal
from bondnet.layer.utils import UnifySize
from ..utils import make_hetero_CH2O


def test_select_not_equal():

    x = torch.tensor(
        [
            [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [6, 7, 8]]],
            [[[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [6, 7, 8]]],
        ]
    )
    y = torch.tensor([[0, 1, 2], [3, 4, 5]])
    target = torch.tensor([[[3, 4, 5], [6, 7, 8]], [[0, 1, 2], [6, 7, 8]]])

    out = select_not_equal(x, y)

    assert torch.equal(out, target)


def test_gated_conv_layer():

    node_types = ["atom", "bond", "global"]

    g, feats = make_hetero_CH2O(self_loop=True)
    num_nodes = {ntype: g.number_of_nodes(ntype) for ntype in node_types}

    unifier = UnifySize(input_dim={"atom": 2, "bond": 3, "global": 4}, output_dim=10)
    feats = unifier(feats)

    out_dim = 20
    gat_layer = GatedGCNConv(input_dim=10, output_dim=out_dim, num_fc_layers=2)
    out = gat_layer(g, feats)

    assert set(out.keys()) == set(node_types)
    for k, v in out.items():
        assert np.array_equal(v.shape, (num_nodes[k], out_dim))
