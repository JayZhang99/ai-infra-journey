from pathlib import Path
import torch

class TinyMLP(torch.nn.Module):
    def __init__(self, in_dim=8, hidden=32, out_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def make_data(samples=512, in_dim=8, out_dim=2, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, in_dim, generator=g)
    w = torch.randn(in_dim, out_dim, generator=g)
    b = torch.randn(out_dim, generator=g)
    noise = 0.01 * torch.randn(
        samples, out_dim, generator=g
    )
    return x, x @ w + b + noise 

def train_model(model, x, target, steps=200, learning_rate=1e-2):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    losses = []

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses

def save_and_reload(model, path: Path):
    torch.save(model.state_dict(), path)
    loaded = TinyMLP()
    state = torch.load(path, map_location="cpu")
    loaded.load_state_dict(state)
    loaded.eval()
    return loaded

def cuda_smoke():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    torch.manual_seed(7)

    x, target = make_data()
    x = x.to(device)
    target = target.to(device)
    model = TinyMLP().to(device)

    losses = train_model(
        model,
        x,
        target
    )
    print(losses[0], losses[-1])

def main():
    torch.manual_seed(7)
    x, target = make_data()
    model = TinyMLP()
    losses = train_model(model, x, target)
    print(losses[0], losses[-1])
    cuda_smoke()


if __name__ == "__main__":
    main()
