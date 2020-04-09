"""
Heterogeneous Graph Attention Networks on molecule level property.
"""

import warnings
import torch.nn as nn
from gnn.layer.gatedconv import GatedGCNConv, GatedGCNConv1
from gnn.layer.readout import Set2SetThenCat
from gnn.layer.utils import UnifySize
from gnn.utils import warn_stdout


class GatedGCNMol(nn.Module):
    """
    Heterograph attention network for molecules.


    Args:
        in_feats (dict): input feature size.
        embedding_size (int): embedding layer size.
        gated_num_layers (int): number of graph attention layer
        gated_hidden_size (list): hidden size of graph attention layers
        gated_num_fc_layers (int):
        gated_graph_norm (bool):
        gated_batch_norm(bool): whether to apply batch norm to gated layer.
        gated_activation (torch activation): activation fn of gated layers
        gated_residual (bool, optional): [description]. Defaults to False.
        gated_dropout (float, optional): dropout ratio for gated layer.
        fc_num_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_batch_norm (bool): whether to apply batch norm to fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_dropout (float, optional): dropout ratio for fc layer.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
    """

    def __init__(
        self,
        in_feats,
        embedding_size=32,
        gated_num_layers=3,
        gated_hidden_size=[64, 64, 32],
        gated_num_fc_layers=1,
        gated_graph_norm=True,
        gated_batch_norm=True,
        gated_activation="ReLU",
        gated_residual=True,
        gated_dropout=0.0,
        num_lstm_iters=5,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        fc_num_layers=2,
        fc_hidden_size=[32, 16],
        fc_batch_norm=False,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=1,
        conv="GatedGCNConv",
    ):
        super(GatedGCNMol, self).__init__()

        if isinstance(gated_activation, str):
            gated_activation = getattr(nn, gated_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        # embedding layer
        self.embedding = UnifySize(in_feats, embedding_size)

        # gated layer
        if conv == "GatedGCNConv":
            conv_fn = GatedGCNConv
        elif conv == "GatedGCNConv1":
            conv_fn = GatedGCNConv1
        else:
            raise ValueError()

        in_size = embedding_size
        self.gated_layers = nn.ModuleList()
        for i in range(gated_num_layers):
            self.gated_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=gated_hidden_size[i],
                    num_fc_layers=gated_num_fc_layers,
                    graph_norm=gated_graph_norm,
                    batch_norm=gated_batch_norm,
                    activation=gated_activation,
                    residual=gated_residual,
                    dropout=gated_dropout,
                )
            )
            in_size = gated_hidden_size[i]

        # set2set readout layer
        ntypes = ["atom", "bond"]
        in_size = [gated_hidden_size[-1]] * len(ntypes)

        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size  of in feature)
        readout_out_size = gated_hidden_size[-1] * 2 + gated_hidden_size[-1] * 2
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gated_hidden_size[-1] * len(set2set_ntypes_direct)

        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            warnings.showwarning = warn_stdout
            warnings.warn(
                f"`fc_dropout = {fc_dropout}` provided for {self.__class__.__name__} "
                f"smaller than {delta}. Ignored."
            )
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size
        for i in range(fc_num_layers):
            out_size = fc_hidden_size[i]
            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_dropout))

            in_size = out_size

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))

    def forward(self, graph, feats, norm_atom, norm_bond):
        """
        Args:
            feats (dict)
            norm_atom (2D tensor)
            norm_bond (2D tensor)

        Returns:
            2D tensor: of shape (N, ft_size)
        """

        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # readout layer
        feats = self.readout_layer(graph, feats)

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats
