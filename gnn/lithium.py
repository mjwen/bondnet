# pylint: disable=no-member
import time
import torch
from gnn.data.dataset import ElectrolyteDataset, train_validation_test_split
from gnn.data.dataloader import DataLoader
from gnn.model.hgat import HGAT
from gnn.metric import MSELoss, MAELoss, evaluate, EarlyStopping
from gnn.args import create_parser

torch.manual_seed(35)

args = create_parser()
if args.gpu > 0 and torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = None

# dataset
sdf_file = "/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf"
label_file = "/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt"
dataset = ElectrolyteDataset(sdf_file, label_file)
trainset, valset, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)
train_loader = DataLoader(trainset, batch_size=10, shuffle=True)

# model
attn_mechanism = {
    "atom": {"edges": ["b2a", "g2a"], "nodes": ["bond", "global"]},
    "bond": {"edges": ["a2b", "g2b"], "nodes": ["atom", "global"]},
    "global": {"edges": ["a2g", "b2g"], "nodes": ["atom", "bond"]},
}
attn_order = ["atom", "bond", "global"]
in_feats = trainset.get_feature_size(attn_order)
model = HGAT(attn_mechanism, attn_order, in_feats)
print(model)
if args.device is not None:
    model.to(device=args.device)

# accuracy metric and stopper
metric = MAELoss()
stopper = EarlyStopping(patience=100)

# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = MSELoss()


for epoch in range(10):
    t0 = time.time()
    model.train()

    epoch_loss = 0
    epoch_pred = []
    epoch_energy = []
    epoch_indicator = []
    for it, (bg, label) in enumerate(train_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in attn_order}
        if args.device is not None:
            feats = {k: v.to(device=args.device) for k, v in feats.items()}
            label = {k: v.to(device=args.device) for k, v in label.items()}
        pred = model(bg, feats)
        loss = loss_func(pred, label["energies"], label["indicators"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        # NOTE keep track of them for later accuracy evaluate. requires a lot of memory.
        # Alternatively, we can compute them later if memory is an issue. Should provide
        # a switch to choose between the two
        epoch_pred.append(pred)
        epoch_energy.append(label["energies"])
        epoch_indicator.append(label["indicators"])

    epoch_loss /= it + 1

    # evaluate the accuracy
    with torch.no_grad():
        train_acc = metric(
            torch.cat(epoch_pred), torch.cat(epoch_energy), torch.cat(epoch_indicator)
        )
    val_acc = evaluate(model, valset, metric, attn_order, args.device)
    if stopper.step(val_acc, model):
        break

    tt = time.time() - t0

    print(
        "Epoch {:4d} | Loss {:12.6e} | TrainAcc {:12.6e} | ValAcc {:12.6e} | "
        "Time (s) {:.4f}".format(epoch, epoch_loss, train_acc, val_acc, tt)
    )

test_acc = evaluate(model, testset, metric, attn_order, args.device)
print("TestAcc {:12.6e}".format(test_acc))
