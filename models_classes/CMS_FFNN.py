import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class MultiTimescaleStack:
    def __init__(self, d_in=128, d_hidden=128, d_out=10):
        # Models
        self.mlp0 = MLP(d_in, d_hidden, d_hidden)
        self.mlp1 = MLP(d_hidden, d_hidden, d_hidden)
        self.mlp2 = MLP(d_hidden, d_hidden, d_hidden)
        self.mlp3 = MLP(d_hidden, d_hidden, d_out)

        # Optimizers (different learning rates)
        self.opt0 = optim.Adam(self.mlp0.parameters(), lr=1e-3)
        self.opt1 = optim.Adam(self.mlp1.parameters(), lr=3e-4)
        self.opt2 = optim.Adam(self.mlp2.parameters(), lr=1e-4)
        self.opt3 = optim.Adam(self.mlp3.parameters(), lr=3e-5)

        # Step counter
        self.step = 0

        # EMA states
        self.h0_ema = None
        self.h1_ema = None
        self.h2_ema = None

    def ema(self, prev, x, beta):
        if prev is None:
            return x
        return beta * prev + (1 - beta) * x

    def forward(self, x):
        # Fast layer
        h0 = self.mlp0(x)
        self.h0_ema = self.ema(self.h0_ema, h0, beta=0.9)

        h0_in = 0.5 * h0 + 0.5 * self.h0_ema

        # Medium layer 1
        h1 = self.mlp1(h0_in)
        self.h1_ema = self.ema(self.h1_ema, h1, beta=0.97)

        h1_in = 0.5 * h1 + 0.5 * self.h1_ema

        # Medium layer 2
        h2 = self.mlp2(h1_in)
        self.h2_ema = self.ema(self.h2_ema, h2, beta=0.99)

        # Slow output layer
        out = self.mlp3(self.h2_ema)

        return out

    def train_step(self, x, y):
        self.step += 1

        # Forward pass
        out = self.forward(x)

        # Loss
        loss = F.cross_entropy(out, y)

        # Backward (single pass)
        loss.backward()

        # Always update fast layer
        self.opt0.step()

        # Slower update schedules (4× scaling)
        if self.step % 4 == 0:
            self.opt1.step()

        if self.step % 16 == 0:
            self.opt2.step()

        if self.step % 64 == 0:
            self.opt3.step()

        # Zero gradients for all optimizers
        self.opt0.zero_grad()
        self.opt1.zero_grad()
        self.opt2.zero_grad()
        self.opt3.zero_grad()

        return loss.item()