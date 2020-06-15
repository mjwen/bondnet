import torch
from collections import OrderedDict


def clean(inname="checkpoint.pkl", outname="checkpoint_corrected.pkl"):
    """
    The state_dict of nn.DataDistributedParallel (DDP) is stored in `module`, while that
    of a regular model does not do this. So, when loading the checkpoint saved by DDP
    to a regular model, there would be error like:
    `KeyError: unexpected key "module.encoder.embedding.weight" in state_dict`
    This function removes the `module.` in the key such that it can be loaded in a
    regular model.

    More info can be found at:
    https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4

    Args:
        inname (str): checkpoint file saved by DDP
        outname (str): name of corrected checkpoint
    """

    ckp = torch.load(inname, map_location=torch.device("cpu"))

    new_ckp = {}

    for k, v in ckp.items():

        # state_dict of the model
        if k == "model":
            d = OrderedDict()
            for kk, vv in v.items():
                kk = kk[7:]  # remove `module.`
                d[kk] = vv
            new_ckp[k] = d

        # other info we saved in the checkpoint does not need to be modified
        else:
            new_ckp[k] = v

    torch.save(new_ckp, outname)


if __name__ == "__main__":
    clean()
