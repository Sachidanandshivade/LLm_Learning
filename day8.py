import torch
import torch.nn as nn

text="hello world"
chars = sorted(list(set(text)))
char_to_id = {ch: i for i, ch in enumerate(chars)}
vocab_size = len(chars)

print("Vocabulary:", char_to_id)
print("Vocab size:", vocab_size)

embedding_dim = 8
embedding = nn.Embedding(vocab_size, embedding_dim)

print("\n=== Embedding Layer ===")
print(f"Turns {vocab_size} token IDs into vectors of size {embedding_dim}")

encoded = torch.tensor([char_to_id[ch] for ch in "hello"])
print("\nEncoded 'hello':", encoded)

embedded = embedding(encoded)
print("Embedded shape:", embedded.shape)  # [5, 8] — 5 chars, 8 numbers each
print("Embedded 'hello':\n", embedded)

print("\n'h' vector:", embedded[0])
print("'e' vector:", embedded[1])
print("'l' vector:", embedded[2])