import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's build self-attention from scratch
# Input: a sentence of 4 tokens, each with embedding size 8
torch.manual_seed(42)  # so we all get same random numbers

seq_len = 4       # 4 tokens in our sentence
embed_dim = 8     # each token is an 8-dimensional vector
head_dim = 8      # size of Q, K, V vectors

# Fake input — pretend these are our embedded tokens
# Shape: [seq_len, embed_dim] = [4, 8]
x = torch.randn(seq_len, embed_dim)
print("Input shape:", x.shape)
print("Input (our 4 embedded tokens):\n", x)

# Step 1 — Create Q, K, V weight matrices
# These are LEARNED during training
W_q = nn.Linear(embed_dim, head_dim, bias=False)
W_k = nn.Linear(embed_dim, head_dim, bias=False)
W_v = nn.Linear(embed_dim, head_dim, bias=False)

# Step 2 — Compute Q, K, V for every token
Q = W_q(x)  # [4, 8] — what each token is looking for
K = W_k(x)  # [4, 8] — what each token contains
V = W_v(x)  # [4, 8] — what each token will give

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Step 3 — Compute attention scores
# How much should each token attend to every other token?
# scores[i][j] = how much token i should attend to token j
scores = torch.matmul(Q, K.transpose(0, 1))
print("\nAttention scores shape:", scores.shape)  # [4, 4]
print("Raw scores:\n", scores)

# Step 4 — Scale the scores
# Divide by sqrt(head_dim) to keep numbers stable
scores = scores / (head_dim ** 0.5)
print("\nScaled scores:\n", scores)

# Step 5 — Softmax — turn scores into probabilities (0 to 1, sum to 1)
weights = F.softmax(scores, dim=-1)
print("\nAttention weights (after softmax):\n", weights)
print("Each row sums to:", weights.sum(dim=-1))

# Step 6 — Weighted sum of Values
output = torch.matmul(weights, V)
print("\nOutput shape:", output.shape)  # [4, 8]
print("Output:\n", output)

print("\n✅ Self-attention complete!")
print("Each token has now looked at every other token")
print("and gathered information based on relevance!")

print("\n\n=== Causal Masking (Making it GPT-like) ===")

# Create a mask — upper triangle is True (these are future tokens)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print("Mask (True = future token, hide these):")
print(mask)

# Apply mask — set future positions to -infinity
# After softmax, -inf becomes 0 (no attention to future)
masked_scores = scores.masked_fill(mask, float('-inf'))
print("\nMasked scores:")
print(masked_scores)

# Softmax again
masked_weights = F.softmax(masked_scores, dim=-1)
print("\nMasked attention weights:")
print(masked_weights)

print("\nToken 0 only sees itself:")
print("weights[0]:", masked_weights[0])
print("\nToken 1 sees tokens 0 and 1:")
print("weights[1]:", masked_weights[1])
print("\nToken 3 sees all tokens:")
print("weights[3]:", masked_weights[3])