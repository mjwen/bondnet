import torch
from gnn.data.dataset import ElectrolyteDataset
from gnn.data.dataloader import DataLoader
from gnn.model.hgat import HGAT
from gnn.loss import MSELoss

torch.manual_seed(35)

sdf_file = "/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_1.sdf"
label_file = "/Users/mjwen/Applications/mongo_db_access/extracted_data/label_1.txt"
dataset = ElectrolyteDataset(sdf_file, label_file)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


attn_mechanism = {
    "atom": {"edges": ["b2a", "g2a"], "nodes": ["bond", "global"]},
    "bond": {"edges": ["a2b", "g2b"], "nodes": ["atom", "global"]},
    "global": {"edges": ["a2g", "b2g"], "nodes": ["atom", "bond"]},
}
attn_order = ["atom", "bond", "global"]
in_feats = dataset.get_feature_size(attn_order)
model = HGAT(attn_mechanism, attn_order, in_feats)
model.train()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = MSELoss(scale=10)

epoch_losses = []
for epoch in range(10):
    epoch_loss = 0
    for it, (bg, label) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in attn_order}
        prediction = model(bg, feats)
        loss = loss_func(prediction, label["energies"], label["indicators"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= it + 1
    print("Epoch {}, loss {:.8f}".format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
