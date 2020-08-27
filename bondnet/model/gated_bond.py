import logging
import torch
import torch.nn as nn
from bondnet.layer.gatedconv import GatedGCNConv
from bondnet.layer.readout import ConcatenateMeanMax, ConcatenateMeanAbsDiff
from bondnet.layer.utils import UnifySize

logger = logging.getLogger(__name__)


class GatedGCNBond(nn.Module):
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
        readout_type="bond",
        fc_num_layers=2,
        fc_hidden_size=[32, 16],
        fc_batch_norm=False,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=1,
        classification=False,
    ):
        super().__init__()

        # assert input
        self.classification = classification
        if not self.classification:
            assert outdim == 1, (
                f"outdim ({outdim}) should be 1 for regression in "
                f"{self.__class__.__name__}"
            )

        # activation fn
        if isinstance(gated_activation, str):
            gated_activation = getattr(nn, gated_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        # embedding layer
        self.embedding = UnifySize(in_feats, embedding_size)

        # gated layer
        in_size = embedding_size
        self.gated_layers = nn.ModuleList()
        for i in range(gated_num_layers):
            self.gated_layers.append(
                GatedGCNConv(
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

        if readout_type == "bond":
            self.readout_layer = lambda graph, feats: feats  # similar to nn.Identity()
            readout_out_size = gated_hidden_size[-1]

        elif readout_type == "bond_cat_mean_max":
            etypes = [("atom", "a2b", "bond")]
            self.readout_layer = ConcatenateMeanMax(etypes=etypes)
            # 3 because we concatenate atom feats to bond feats  in the readout_layer
            readout_out_size = gated_hidden_size[-1] * 3

        elif readout_type == "bond_cat_mean_diff":
            etypes = [("atom", "a2b", "bond")]
            self.readout_layer = ConcatenateMeanAbsDiff(etypes=etypes)
            # 3 because we concatenate atom feats to bond feats  in the readout_layer
            readout_out_size = gated_hidden_size[-1] * 3
        else:
            raise ValueError("readout_type='{}' not supported.".format(readout_type))

        # need dropout?
        delta = 1e-3
        if fc_dropout < delta:
            logger.warning(
                "`fc_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(fc_dropout, self.__class__.__name__, delta)
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

    def forward(self, graph, feats, norm_atom, norm_bond, mol_based=False):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            norm_atom (2D tensor)
            norm_bond (2D tensor)
            mol_based (bool): If `True`, split the predictions for bonds to the
                predicitons of molecules based on the number of bonds in each molecule.
                This determines the shape of the return value.
        Returns:
            list of 2D Tensor: bond class scores. If `classification` is `True` and if
                `mol_based` is `True`. Each tensor corresponds to a molecule.
            2D Tensor: bond class scores. If `classification` is `True` and if
                `mol_based` is `False`.
            2D Tensor: molecule energies. If not `classification` and if `mol_based` is
                `True`.
            1D Tensor: bond energies. If not `classification` and if `mol_based` is
                `False`.
        """

        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # readout layer for bond features only
        feats = self.readout_layer(graph, feats)
        feats = feats["bond"]

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        if self.classification:
            if mol_based:
                # list of 2D tensor of shape (N, outdim); N is the number of bonds in each
                # graph and it could be different from graph to graph
                res = self._split_batched_output(graph, feats)
            else:
                # 2D tensor of shape (Nb, outdim); Nb is the total number of bonds in
                # all graphs
                res = feats
        else:
            if mol_based:
                # 2D tensor of shape (Nm,1); Nm is the number graphs
                res = feats.view(-1)
                res = self._bond_pred_to_mol_pred(graph, res)
            else:
                # 1D tensor (Nb,); Nb is the total number of bonds in all graphs
                res = feats.view(-1)

        return res

    def feature_before_fc(self, graph, feats):
        """
        This is used when we want to visualize feature.
        """
        # hgat layer
        for i, layer in enumerate(self.gat_layers):
            feats = layer(graph, feats)

            # apply activation after average over heads (eq. 6 of the GAT paper)
            # see below in forward()
            if i == len(self.gat_layers) - 1:
                for nt in feats:
                    ft = feats[nt].view(feats[nt].shape[0], self.num_heads, -1)
                    feats[nt] = self.gat_activation(torch.mean(ft, dim=1))

        # readout layer
        feats = self.readout_layer(graph, feats)
        res = feats["bond"]

        return res

    @staticmethod
    def _split_batched_output(graph, value):
        """
        Split a tensor into `num_graphs` chunks, the size of each chunk equals the
        number of bonds in the graph.

        Returns:
            list of tensor.

        """
        nbonds = graph.batch_num_nodes("bond")
        return torch.split(value, nbonds)

    def _bond_pred_to_mol_pred(self, graph, bond_pred):
        """
        Sum the bond predictions in each molecule to get molecule predictions.
        """
        bond_pred = self._split_batched_output(graph, bond_pred)
        mol_pred = torch.stack([torch.sum(i) for i in bond_pred]).view((-1, 1))
        return mol_pred
