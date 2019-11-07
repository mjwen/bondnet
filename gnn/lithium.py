import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from dgl.model_zoo.chem import MPNNModel
import networkx as nx
import matplotlib.pyplot as plt
from gnn.data.dataset import ElectrolyteDataset

##########################################################################################
# dataset
##########################################################################################
dataset = ElectrolyteDataset(sdf_file="./electrolyte.sdf", label_file="./electrolyte.csv")
print("dataset size:", len(dataset))
for g, label in dataset:
    print("graph", g)
    for k, v in g.ndata.items():
        print(k, v, v.shape)
    for k, v in g.edata.items():
        print(k, v, v.shape)
    print("label", label, label.shape)

    nx_G = g.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[0.7, 0.7, 0.7]])
    # plt.show()

    break

print("feature_size", dataset.feature_size)

# batch of data
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # NOTE we should be able to use `torch.tensor(lables)` directly
    # this is a torch bug, expected to be fixed
    labels = [l.numpy() for l in labels]
    labels = torch.tensor(labels)
    return batched_graph, labels


data_loader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=collate)


##########################################################################################
model = MPNNModel(
    node_input_dim=13,
    edge_input_dim=5,
    output_dim=1,
    node_hidden_dim=64,
    edge_hidden_dim=128,
    num_step_message_passing=3,
    num_step_set2set=6,
    num_layer_set2set=3,
)
model.train()


##########################################################################################
# training
##########################################################################################
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_losses = []
for epoch in range(30):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        n_feat = bg.ndata["n_feat"]
        e_feat = bg.edata["e_feat"]
        prediction = model(bg, n_feat, e_feat)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= iter + 1
    print("Epoch {}, loss {:.4f}".format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
