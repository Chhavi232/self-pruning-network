import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# setting seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# -----------------------------------------------------------
# Part 1 - Custom PrunableLinear Layer
# -----------------------------------------------------------
# The idea here is simple: each weight gets a "gate" scalar.
# gate = sigmoid(gate_score), so it's always between 0 and 1.
# effective weight = weight * gate
# if gate -> 0, the weight is basically dead / pruned.
# gradients flow through both weight and gate_scores via autograd.

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # normal weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # gate scores - same shape as weight, learned during training
        # initializing to 0 so sigmoid gives 0.5 initially (all gates half open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # kaiming init for weights (standard practice)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        # squish gate scores to (0,1)
        gates = torch.sigmoid(self.gate_scores)

        # multiply weights elementwise with gates
        w = self.weight * gates

        return F.linear(x, w, self.bias)

    def get_gates(self):
        # helper to pull gate values out for analysis
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def sparsity_loss(self):
        # L1 norm of gates = just sum of gate values (since they're all positive)
        return torch.sigmoid(self.gate_scores).sum()


# -----------------------------------------------------------
# Network
# -----------------------------------------------------------
# simple feedforward, nothing fancy
# using batchnorm + dropout to help with training stability

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = PrunableLinear(128, 10)

        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the image

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)

        x = F.relu(self.bn3(self.fc3(x)))

        x = self.fc4(x)
        return x

    def total_sparsity_loss(self):
        loss = torch.tensor(0.0).to(device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                loss += m.sparsity_loss()
        return loss

    def sparsity_level(self, threshold=1e-2):
        # % of gates that are basically zero
        all_gates = self.all_gate_values()
        pruned = (all_gates < threshold).sum()
        return pruned / len(all_gates)

    def all_gate_values(self):
        g = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g.append(m.get_gates().flatten().numpy())
        return np.concatenate(g)


# -----------------------------------------------------------
# Data
# -----------------------------------------------------------
def load_data(batch_size=128):
    # standard cifar10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.247,  0.243,  0.261)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# -----------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------
def train_epoch(model, loader, optimizer, lam):
    model.train()
    total_loss = 0
    correct = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)

        # total loss = classification loss + sparsity regularization
        # lambda controls how hard we push toward pruning
        ce   = F.cross_entropy(out, labels)
        sp   = model.total_sparsity_loss()
        loss = ce + lam * sp

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


def run(lam, train_loader, test_loader, epochs=30):
    print(f"\n--- lambda = {lam} ---")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, lam)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            sp = model.sparsity_level()
            print(f"  ep {epoch:2d} | loss {tr_loss:.3f} | train acc {tr_acc*100:.1f}% | sparsity {sp*100:.1f}%")

    test_acc = evaluate(model, test_loader)
    sparsity = model.sparsity_level()

    print(f"  => test acc: {test_acc*100:.2f}%  |  sparsity: {sparsity*100:.2f}%")

    return {
        "lam":      lam,
        "test_acc": test_acc,
        "sparsity": float(sparsity),
        "gates":    model.all_gate_values(),
    }


# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------
def plot_gates(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    colors = ["steelblue", "seagreen", "tomato"]

    for ax, res, c in zip(axes, results, colors):
        ax.hist(res["gates"], bins=60, color=c, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(0.01, color="black", linestyle="--", linewidth=1, label="threshold=0.01")
        ax.set_title(
            f"λ={res['lam']}\nacc={res['test_acc']*100:.1f}%  sparsity={res['sparsity']*100:.1f}%",
            fontsize=10
        )
        ax.set_xlabel("gate value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    plt.suptitle("Gate value distributions after training", fontsize=12)
    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150)
    print("\nplot saved to gate_distributions.png")
    plt.close()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    EPOCHS     = 30
    BATCH_SIZE = 128
    LAMBDAS    = [1e-5, 1e-4, 1e-3]

    train_loader, test_loader = load_data(BATCH_SIZE)

    all_results = []
    for lam in LAMBDAS:
        res = run(lam, train_loader, test_loader, epochs=EPOCHS)
        all_results.append(res)

    # print summary
    print("\n\nSummary")
    print(f"{'Lambda':<12} {'Test Acc':>10} {'Sparsity':>12}")
    print("-" * 36)
    for r in all_results:
        print(f"{r['lam']:<12} {r['test_acc']*100:>9.2f}%  {r['sparsity']*100:>10.2f}%")

    plot_gates(all_results)
