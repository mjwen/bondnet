import torch
from bondnet.layer.utils import UnifySize


def test_unify_size():
    in_feats = {"a": 2, "b": 3}
    out_feats = 4
    us = UnifySize(in_feats, out_feats)

    feats = {"a": torch.zeros(2), "b": torch.zeros(3)}
    feats = us(feats)
    for _, v in feats.items():
        assert v.shape[0] == out_feats
