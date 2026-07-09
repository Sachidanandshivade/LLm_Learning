# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# torch.manual_seed(42)
#
# # ============================================
# # Training text — we'll use a simple paragraph
# # ============================================
# text = """
# Name: Sachidanand Shivade
# Title: Software Developer / Full Stack Developer
# Professional Summary: Full Stack Developer skilled in React, Spring Boot, Hibernate, MySQL, and MongoDB, with a focus on building robust and scalable web applications. Eager to leverage technical foundation to develop innovative solutions that drive business growth and deliver exceptional user experiences. Looking to contribute meaningfully to a forward-thinking team while continuously growing as an engineer.
# Contact:
# Phone: 9353947020
# Email: shivade.sachidanand@gmail.com
# LinkedIn: sachidanand-shivade-197a802a0
# GitHub: github.com/Sachidanandshivade
# Education:
# Reva University, 2022-2026, Bachelor of Technology (CSIT), CGPA 8.6
# Sri Mate Manikeshwari PU College, 2020-2022, Pre-University Science, Percentage 85%
# Internship:
# Software Development Intern, Kodnest Technologies, Bengaluru, Karnataka, India, 23 Mar 2026 - Present
#
# Developed proficiency in Java, MySQL, frontend development and backend development (Spring Boot)
# Acquired practical experience with real-world development workflows and debugging processes
# Collaborated on team-based projects, enhancing problem-solving skills and ability to work effectively in a team setting
#
# """
#
# # ============================================
# # Tokenizer
# # ============================================
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# char_to_id = {ch: i for i, ch in enumerate(chars)}
# id_to_char = {i: ch for i, ch in enumerate(chars)}
#
# def encode(s):
#     return [char_to_id[ch] for ch in s]
#
# def decode(ids):
#     return "".join([id_to_char[i] for i in ids])
#
# print(f"Text length: {len(text)} characters")
# print(f"Vocab size: {vocab_size}")
# print(f"Vocabulary: {''.join(chars)}")
#
# # Encode entire text
# data = torch.tensor(encode(text))
# print(f"Encoded data shape: {data.shape}")
#
# # ============================================
# # Training data — input/target pairs
# # ============================================
# block_size = 32  # how many characters the model sees at once
#
# def get_batch(data, block_size, batch_size=4):
#     # Pick random starting positions
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     # Input = characters at positions ix to ix+block_size
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     # Target = same but shifted by 1 (next character)
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     return x, y
#
# x, y = get_batch(data, block_size)
# print(f"\nSample batch:")
# print(f"Input shape:  {x.shape}")   # [4, 32]
# print(f"Target shape: {y.shape}")   # [4, 32]
# print(f"Input text:   '{decode(x[0].tolist())}'")
# print(f"Target text:  '{decode(y[0].tolist())}'")
#
# # ============================================
# # GPT Model (from Day 11)
# # ============================================
# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.out = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, x):
#         seq_len = x.shape[1]
#         Q = self.W_q(x)
#         K = self.W_k(x)
#         V = self.W_v(x)
#         scores = torch.matmul(Q, K.transpose(-2, -1))
#         scores = scores / (self.embed_dim ** 0.5)
#         mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
#         scores = scores.masked_fill(mask, float('-inf'))
#         weights = F.softmax(scores, dim=-1)
#         return self.out(torch.matmul(weights, V))
#
#
# class FeedForward(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.ReLU(),
#             nn.Linear(embed_dim * 4, embed_dim)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.attention = SelfAttention(embed_dim)
#         self.ff = FeedForward(embed_dim)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#
#     def forward(self, x):
#         x = self.norm1(x + self.attention(x))
#         x = self.norm2(x + self.ff(x))
#         return x
#
#
# class GPT(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_blocks, max_seq_len):
#         super().__init__()
#         self.token_emb = nn.Embedding(vocab_size, embed_dim)
#         self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
#         self.blocks = nn.Sequential(
#             *[TransformerBlock(embed_dim) for _ in range(num_blocks)]
#         )
#         self.norm = nn.LayerNorm(embed_dim)
#         self.lm_head = nn.Linear(embed_dim, vocab_size)
#
#     def forward(self, token_ids):
#         seq_len = token_ids.shape[1]
#         tok = self.token_emb(token_ids)
#         positions = torch.arange(seq_len)
#         pos = self.pos_emb(positions)
#         x = tok + pos
#         x = self.blocks(x)
#         x = self.norm(x)
#         return self.lm_head(x)
#
#
# # ============================================
# # Training Loop
# # ============================================
# embed_dim   = 64
# num_blocks  = 3
# max_seq_len = block_size
#
# model = GPT(vocab_size, embed_dim, num_blocks, max_seq_len)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"\n=== GPT Model ===")
# print(f"Parameters: {total_params:,}")
#
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
# print(f"\n=== Training ===")
# for step in range(5000):
#     # Get random batch
#     x, y = get_batch(data, block_size)
#
#     # Forward pass
#     logits = model(x)   # [batch, seq_len, vocab_size]
#
#     # Compute loss
#     # Reshape for cross entropy: [batch*seq_len, vocab_size] vs [batch*seq_len]
#     loss = F.cross_entropy(
#         logits.view(-1, vocab_size),
#         y.view(-1)
#     )
#
#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if step % 300 == 0:
#         print(f"Step {step:4d}: loss = {loss.item():.4f}")
#
# print("\n✅ Training complete!")
#
# # ============================================
# # Text Generation
# # ============================================
# def generate(model, start_text, max_new_tokens=200):
#     model.eval()
#     token_ids = torch.tensor(encode(start_text)).unsqueeze(0)  # [1, seq_len]
#
#     generated = start_text
#     for _ in range(max_new_tokens):
#         # Only use last block_size tokens
#         input_ids = token_ids[:, -block_size:]
#
#         # Get predictions
#         with torch.no_grad():
#             logits = model(input_ids)
#
#         # Focus on last token's prediction
#         last_logits = logits[0, -1, :]
#
#         # Convert to probabilities
#         probs = F.softmax(last_logits, dim=-1)
#
#         # Sample next token
#         next_id = torch.multinomial(probs, num_samples=1).item()
#
#         # Append to sequence
#         next_char = id_to_char[next_id]
#         generated += next_char
#         token_ids = torch.cat([
#             token_ids,
#             torch.tensor([[next_id]])
#         ], dim=1)
#
#     return generated
#
# print("\n=== Text Generation ===")
# print("\nStarting with: 'Sachidanand'")
# print("-" * 40)
# print(generate(model, "Sachidanand", max_new_tokens=250))
# print("-" * 40)
#
# print("\nStarting with: 'He knows'")
# print("-" * 40)
# print(generate(model, "Skills", max_new_tokens=250))
# print("-" * 40)