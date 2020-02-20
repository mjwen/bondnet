"""
Graph Attention Networks for heterograph.
"""

import torch
import torch.nn as nn
from gnn.layer.hgatconv import HGATConv
from gnn.layer.readout import ConcatenateMeanMax, ConcatenateMeanAbsDiff
from dgl import BatchedDGLHeteroGraph
import warnings
from gnn.utils import warn_stdout


class HGAT(nn.Module):
    """
    Heterograph attention network.


    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key,
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): The order to attend the node features. Order matters.
        in_feats (list): input feature size for the corresponding node in `attn_order`.
        num_gat_layers (int): number of graph attention layer
        gat_hidden_size (list): hidden size of graph attention layers
        gat_activation (torch activation): activation fn of gat layers
        num_heads (int): number of attention heads, the same for all nodes
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        residual (bool, optional): [description]. Defaults to False.
        num_fc_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_activation (torch activation): activation fn of fc layers
        fc_drop (float, optional): dropout ratio for fc layer.
        readout_type (str): how to read out the features to bonds and then pass the fc
            layers. Options are {'bond', 'bond_cat_mean_max', 'bond_cat_mean_diff'}.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        num_gat_layers=3,
        gat_hidden_size=[32, 64, 128],
        gat_activation="ELU",
        num_heads=4,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        num_fc_layers=3,
        fc_hidden_size=[128, 64, 32],
        fc_activation="ELU",
        fc_drop=0.0,
        readout_type="bond",
        outdim=1,
    ):
        super(HGAT, self).__init__()

        self.outdim = outdim

        # activation fn
        if isinstance(gat_activation, str):
            gat_activation = getattr(nn, gat_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        self.gat_layers = nn.ModuleList()

        # input projection (no dropout)
        self.gat_layers.append(
            HGATConv(
                attn_mechanism=attn_mechanism,
                attn_order=attn_order,
                in_feats=in_feats,
                out_feats=gat_hidden_size[0],
                num_heads=num_heads,
                feat_drop=0.0,
                attn_drop=0.0,
                negative_slope=negative_slope,
                residual=residual,
                activation=gat_activation,
            )
        )

        # hidden gat layers
        for i in range(1, num_gat_layers):
            in_size = [gat_hidden_size[i - 1] * num_heads for _ in in_feats]
            self.gat_layers.append(
                HGATConv(
                    attn_mechanism=attn_mechanism,
                    attn_order=attn_order,
                    in_feats=in_size,
                    out_feats=gat_hidden_size[i],
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=gat_activation,
                )
            )

        # TODO to be general, this could and should be passed in as argument
        if readout_type == "bond":
            self.readout_layer = lambda graph, feats: feats  # similar to nn.Identity()
            readout_out_size = gat_hidden_size[-1] * num_heads

        elif readout_type == "bond_cat_mean_max":
            etypes = [("atom", "a2b", "bond")]
            self.readout_layer = ConcatenateMeanMax(etypes=etypes)
            # 3 because we concatenate atom feats to bond feats  in the readout_layer
            readout_out_size = gat_hidden_size[-1] * num_heads * 3

        elif readout_type == "bond_cat_mean_diff":
            etypes = [("atom", "a2b", "bond")]
            self.readout_layer = ConcatenateMeanAbsDiff(etypes=etypes)
            # 3 because we concatenate atom feats to bond feats  in the readout_layer
            readout_out_size = gat_hidden_size[-1] * num_heads * 3
        else:
            raise ValueError("readout_type='{}' not supported.".format(readout_type))

        # need dropout?
        delta = 1e-3
        if fc_drop < delta:
            warnings.showwarning = warn_stdout
            warnings.warn(
                "`fc_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(feat_drop, self.__class__.__name__, delta)
            )
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size
        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(in_size, fc_hidden_size[i]))
            self.fc_layers.append(fc_activation)
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_drop))

            in_size = fc_hidden_size[i]

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, self.outdim))

    def forward(self, graph, feats, mol_energy=False):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            mol_energy (bool): If `True`, sum the prediction of bond energies as
                molecule energy.
        Returns:
            1D Tensor: bond energies. If `mol_energy` is `True`, then return a 2D
                tensor of shape (N, 1), where `N` is the number of molecules in the
                batch of data.
            list of 2D tensor: if classification if `True`.
        """

        # hgat layer
        for layer in self.gat_layers:
            feats = layer(graph, feats)

        # readout layer for bond features only
        feats = self.readout_layer(graph, feats)
        feats = feats["bond"]

        # TODO to be general, this could and should be passed in as argument
        # fc, activation, and dropout
        for layer in self.fc_layers:
            feats = layer(feats)

        if self.outdim == 1:  # regression
            res = feats.view(-1)  # 1D tensor (N,)
            if mol_energy:
                res = self._bond_energy_to_mol_energy(graph, res)  # 2D tensor (N, 1)
        else:
            res = self._split_batched_output(graph, feats)  # list of 2D tensor

        return res

    @staticmethod
    def _split_batched_output(graph, value):
        """
        Split a tensor into `num_graphs` chunks, the size of each chunk equals the
        number of bonds in the graph.

        Returns:
            list of tensor.

        """
        if isinstance(graph, BatchedDGLHeteroGraph):
            nbonds = graph.batch_num_nodes("bond")
            return torch.split(value, nbonds)
        else:
            return [value]

    def _bond_energy_to_mol_energy(self, graph, bond_energy):
        bond_energy = self._split_batched_output(graph, bond_energy)
        mol_energy = torch.stack([torch.sum(i) for i in bond_energy]).view((-1, 1))
        return mol_energy

    def feature_before_fc(self, graph, feats):
        """
        This is used when we want to visualize feature.
        """
        # hgat layer
        for layer in self.gat_layers:
            feats = layer(graph, feats)

        # readout layer
        feats = self.readout_layer(graph, feats)
        res = feats["bond"]

        return res
