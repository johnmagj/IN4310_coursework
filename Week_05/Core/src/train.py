import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def make_toy_dataloader(batch_size: int = 64, n: int = 10000):
    # Dummy data (no assignment data)
    x = torch.randn(n, 3, 64, 64)
    y = torch.randint(0, 4, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

class TinyModel(nn.Module):
    # Simple placeholder model (not an assignment model)
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--outdir", type=str, default="runs/exp0")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA_VISIBLE_DEVICES:", str(torch.cuda.device_count()), "visible GPU(s)")

    loader = make_toy_dataloader(batch_size=args.batch_size)

    model = TinyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | loss={total_loss/len(loader):.4f}")

    # save weights
    ckpt = outdir / "model.pt"
    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)
    print("Total time (s):", round(time.time() - start, 2))

if __name__ == "__main__":
    main()