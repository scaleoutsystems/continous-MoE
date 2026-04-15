import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Window utilities (TRUE Swin)
# =========================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

# =========================================================
# Window Attention
# =========================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out

# =========================================================
# Prototype Memory with Compression
# =========================================================
class DomainMemory(nn.Module):
    def __init__(self, dim, num_experts, max_prototypes=50, merge_threshold=0.9, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.max_prototypes = max_prototypes
        self.merge_threshold = merge_threshold
        self.momentum = momentum

        self.prototypes = []        # list of [D]
        self.routing_stats = []     # list of [E]

    def _cosine(self, a, b):
        return F.cosine_similarity(a, b, dim=-1)

    def compute_similarity(self, z_x):
        if len(self.prototypes) == 0:
            return None

        sims = []
        for z_d in self.prototypes:
            sims.append(self._cosine(z_x, z_d))

        sims = torch.stack(sims, dim=-1)  # [B, K]
        weights = F.softmax(sims, dim=-1)
        return weights

    def add_or_merge(self, z, routing):
        """
        z: [D]
        routing: [E]
        """
        if len(self.prototypes) == 0:
            self.prototypes.append(z.detach().clone())
            self.routing_stats.append(routing.detach().clone())
            return

        # Find closest prototype
        sims = torch.stack([self._cosine(z, p) for p in self.prototypes])
        best_idx = torch.argmax(sims)
        best_sim = sims[best_idx]

        # Merge if similar
        if best_sim > self.merge_threshold:
            self.prototypes[best_idx] = (
                self.momentum * self.prototypes[best_idx]
                + (1 - self.momentum) * z
            )
            self.routing_stats[best_idx] = (
                self.momentum * self.routing_stats[best_idx]
                + (1 - self.momentum) * routing
            )
        else:
            self.prototypes.append(z.detach().clone())
            self.routing_stats.append(routing.detach().clone())

        # Compression: if too many prototypes, merge closest pair
        if len(self.prototypes) > self.max_prototypes:
            self.compress()

    def compress(self):
        K = len(self.prototypes)
        sims = torch.zeros(K, K)

        for i in range(K):
            for j in range(i+1, K):
                sims[i, j] = self._cosine(self.prototypes[i], self.prototypes[j])

        i, j = torch.nonzero(sims == sims.max())[0]

        # Merge i and j
        new_proto = 0.5 * (self.prototypes[i] + self.prototypes[j])
        new_route = 0.5 * (self.routing_stats[i] + self.routing_stats[j])

        # Remove and replace
        self.prototypes.pop(max(i,j))
        self.prototypes.pop(min(i,j))
        self.routing_stats.pop(max(i,j))
        self.routing_stats.pop(min(i,j))

        self.prototypes.append(new_proto)
        self.routing_stats.append(new_route)

# =========================================================
# Top-2 MoE with similarity-aware routing
# =========================================================
class Top2MoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, memory):
        super().__init__()
        self.num_experts = num_experts
        self.memory = memory

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W

        x_flat = x.view(B * N, C)

        # Compute prototype
        z_x = x.mean(dim=(1,2))  # [B, C]

        weights = self.memory.compute_similarity(z_x)

        # Build routing bias
        if weights is not None:
            routing_bias = torch.zeros(B, self.num_experts, device=x.device)
            for i in range(len(self.memory.prototypes)):
                routing_bias += weights[:, i:i+1] * self.memory.routing_stats[i]

            routing_bias = routing_bias.unsqueeze(1).expand(B, N, self.num_experts)
            routing_bias = routing_bias.reshape(B * N, self.num_experts)
        else:
            routing_bias = torch.zeros(B * N, self.num_experts, device=x.device)

        logits = self.gate(x_flat) + routing_bias
        probs = F.softmax(logits, dim=-1)

        # Top-2 routing
        top2_val, top2_idx = torch.topk(probs, k=2, dim=-1)
        top2_val = top2_val / (top2_val.sum(dim=-1, keepdim=True) + 1e-9)

        outputs = torch.zeros_like(x_flat)
        expert_load = torch.zeros(self.num_experts, device=x.device)

        for i in range(2):
            idx = top2_idx[:, i]
            weight = top2_val[:, i].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.sum() == 0:
                    continue

                selected = x_flat[mask]
                out = self.experts[e](selected)

                outputs[mask] += weight[mask] * out
                expert_load[e] += mask.sum()

        outputs = outputs.view(B, H, W, C)

        # =========================
        # Losses
        # =========================
        probs_reshaped = probs.view(B, N, self.num_experts)

        # Load balancing
        load = expert_load / (B * N + 1e-6)
        prob_mean = probs.mean(dim=0)
        loss_balance = (load * prob_mean).sum() * self.num_experts

        # Consistency
        domain_mean = probs_reshaped.mean(dim=1, keepdim=True)
        loss_consistency = ((probs_reshaped - domain_mean) ** 2).mean()

        # Similarity-based regularization
        loss_reg = 0.0
        if weights is not None:
            for b in range(B):
                p_current = probs_reshaped[b].mean(dim=0)

                p_target = torch.zeros(self.num_experts, device=x.device)
                for i in range(len(self.memory.prototypes)):
                    p_target += weights[b, i] * self.memory.routing_stats[i]

                p_target = p_target / (p_target.sum() + 1e-6)

                loss_reg += F.kl_div(
                    (p_current + 1e-6).log(),
                    p_target + 1e-6,
                    reduction='batchmean'
                )

            loss_reg /= B

        aux_loss = loss_balance + loss_consistency + 0.5 * loss_reg

        # =========================
        # Update memory
        # =========================
        with torch.no_grad():
            routing = probs_reshaped.mean(dim=1)  # [B, E]
            for b in range(B):
                self.memory.add_or_merge(z_x[b], routing[b])

        return outputs, aux_loss

# =========================================================
# Swin Block
# =========================================================
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio, num_experts, memory):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.moe = Top2MoE(dim, int(dim * mlp_ratio), num_experts, memory)

    def forward(self, x):
        B, H, W, C = x.shape

        # Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        # Window attention
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

        attn_windows = self.attn(self.norm1(x_windows))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))

        # MoE
        h = self.norm2(x)
        moe_out, aux_loss = self.moe(h)
        x = x + moe_out

        return x, aux_loss

# =========================================================
# Model
# =========================================================
class Model(nn.Module):
    def __init__(self, dim=96, depth=4, num_heads=4, window_size=7, num_experts=6):
        super().__init__()

        self.memory = DomainMemory(dim, num_experts)

        self.layers = nn.ModuleList([
            SwinBlock(
                dim,
                num_heads,
                window_size,
                0 if i % 2 == 0 else window_size // 2,
                4.0,
                num_experts,
                self.memory
            )
            for i in range(depth)
        ])

        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        total_aux = 0
        for layer in self.layers:
            x, aux = layer(x)
            total_aux += aux

        x = x.mean(dim=(1,2))
        return self.head(x), total_aux

# =========================================================
# Training
# =========================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(500):
        B, H, W, C = 4, 28, 28, 96
        x = torch.randn(B, H, W, C).to(device)
        y = torch.randint(0, 10, (B,), device=device)

        logits, aux_loss = model(x)

        loss = F.cross_entropy(logits, y) + 0.05 * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | Prototypes {len(model.memory.prototypes)}")

if __name__ == "__main__":
    train()