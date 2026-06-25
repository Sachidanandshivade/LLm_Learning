import torch
import torch.nn as nn

# Our vocabulary (from Day 7)
text = "hello world"
chars = sorted(list(set(text)))
char_to_id = {ch: i for i, ch in enumerate(chars)}
vocab_size = len(chars)

print("Vocabulary:", char_to_id)
print("Vocab size:", vocab_size)

# Embedding layer
# vocab_size = how many unique tokens we have
# embedding_dim = how many numbers represent each token
embedding_dim = 8
embedding = nn.Embedding(vocab_size, embedding_dim)

print("\n=== Embedding Layer ===")
print(f"Turns {vocab_size} token IDs into vectors of size {embedding_dim}")

# Encode "hello"
encoded = torch.tensor([char_to_id[ch] for ch in "hello"])
print("\nEncoded 'hello':", encoded)

# Look up embeddings
embedded = embedding(encoded)
print("Embedded shape:", embedded.shape)  # [5, 8] — 5 chars, 8 numbers each
print("Embedded 'hello':\n", embedded)

# Each character is now a vector of 8 numbers
print("\n'h' vector:", embedded[0])
print("'e' vector:", embedded[1])
print("'l' vector:", embedded[2])


print("\n=== Positional Encoding ===")

# The problem — same word, different positions
sentence1 = "hello"
sentence2 = "olleh"  # reversed!

enc1 = torch.tensor([char_to_id[ch] for ch in sentence1])
enc2 = torch.tensor([char_to_id[ch] for ch in sentence2])

emb1 = embedding(enc1)
emb2 = embedding(enc2)

print("Same embeddings for 'hello' and 'olleh'?", torch.equal(emb1, emb2))
# True! The model can't tell them apart without position info

# Positional Encoding — learned version (simplest approach)
max_seq_len = 10  # max sentence length
pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

# Create position indices [0, 1, 2, 3, 4]
positions = torch.arange(len("hello"))
print("\nPositions:", positions)

# Get position embeddings
pos_emb = pos_embedding(positions)
print("Position embeddings shape:", pos_emb.shape)

# Add token embeddings + position embeddings
# This is EXACTLY what GPT does!
final_embedding = emb1 + pos_emb
print("\nFinal embedding (token + position):")
print(final_embedding.shape)
print(final_embedding)

print("\n'h' at position 0:", final_embedding[0])
print("'o' at position 4:", final_embedding[4])
print("\nNow every token knows WHAT it is AND WHERE it is! ✅")
