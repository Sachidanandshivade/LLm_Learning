import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

text = "hello world how are you"
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [char_to_id[ch] for ch in text]

def decode(ids):
    return "".join([id_to_char[i] for i in ids])

print("Vocabulary:", char_to_id)
print("Vocab size:", vocab_size)


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
        scores = torch.matmul(Q, K.transpose(0, 1))
        scores = scores / (self.embed_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return self.out(torch.matmul(weights, V))


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
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
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_blocks, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.shape[0]
        tok = self.token_emb(token_ids)
        positions = torch.arange(seq_len)
        pos = self.pos_emb(positions)
        x = tok + pos
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


embed_dim = 32
num_blocks = 2
max_seq_len = 50

model = GPT(vocab_size, embed_dim, num_blocks, max_seq_len)
total = sum(p.numel() for p in model.parameters())

print(f"\n=== GPT Created ===")
print(f"Vocab size:  {vocab_size}")
print(f"Embed dim:   {embed_dim}")
print(f"Blocks:      {num_blocks}")
print(f"Parameters:  {total}")

sample = "hello"
token_ids = torch.tensor(encode(sample))

print(f"\n=== Forward Pass ===")
print(f"Input text:  '{sample}'")
print(f"Token IDs:   {token_ids}")

logits = model(token_ids)
print(f"Logits shape: {logits.shape}")

probs = F.softmax(logits[-1], dim=-1)
print(f"Probabilities shape: {probs.shape}")
print(f"Sum of probs: {probs.sum().item():.4f}")

next_token_id = torch.argmax(probs).item()
next_char = id_to_char[next_token_id]
print(f"\nMost likely next character after '{sample}': '{next_char}'")
print("\n✅ Complete GPT working end to end!")