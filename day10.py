import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        seq_len = x.shape[0]

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / (self.embed_dim ** 0.5)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Expand to 4x size then back — standard in transformers
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self attention + residual connection + norm
        x = self.norm1(x + self.attention(x))

        # Feed forward + residual connection + norm
        x = self.norm2(x + self.ff(x))
        return x


# Test it!
embed_dim = 8
seq_len = 4

# Fake input — 4 tokens, each 8-dimensional
x = torch.randn(seq_len, embed_dim)
print("Input shape:", x.shape)
print("Input:\n", x)

# Create one transformer block
block = TransformerBlock(embed_dim)

# Pass through
output = block(x)
print("\nOutput shape:", output.shape)
print("Output:\n", output)

print("\n✅ One transformer block complete!")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print("Shape preserved — can stack N of these to make GPT!")

# Count parameters
total_params = sum(p.numel() for p in block.parameters())
print(f"\nParameters in one block: {total_params}")
print(f"GPT-2 has 12 blocks × ~3M params = ~36M total")
print(f"GPT-4 has 96+ blocks × much bigger = ~1.7T params")

print("\n\n=== Stacking Multiple Blocks ===")

# Stack 3 transformer blocks
blocks = nn.Sequential(
    TransformerBlock(embed_dim),
    TransformerBlock(embed_dim),
    TransformerBlock(embed_dim)
)

output = blocks(x)
print("After 3 stacked blocks:")
print("Output shape:", output.shape)

total = sum(p.numel() for p in blocks.parameters())
print(f"Total parameters (3 blocks): {total}")
print(f"That's {total} numbers the model can learn!")

print("\n=== This IS a mini GPT! ===")
print("GPT-2 = these exact blocks × 12, with bigger embed_dim")
print("GPT-3 = these exact blocks × 96, with even bigger embed_dim")
print("GPT-4 = same idea, massively scaled")