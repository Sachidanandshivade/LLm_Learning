import torch
import torch.nn as nn

text = "hello world"
chars = sorted(list(set(text)))
char_to_id = {ch: i for i, ch in enumerate(chars)}

embedding_dim = 4  # small so output is readable
vocab_size = len(chars)

token_emb = nn.Embedding(vocab_size, embedding_dim)
pos_emb = nn.Embedding(10, embedding_dim)

sentence = "hello"
token_ids = torch.tensor([char_to_id[ch] for ch in sentence])
positions = torch.arange(len(sentence))

print("Token IDs:", token_ids)
print("Positions:", positions)

tok = token_emb(token_ids)
pos = pos_emb(positions)
final = tok + pos

print("\n--- Proving position matters ---")
print("'l' token embedding at position 2:", tok[2])
print("'l' token embedding at position 3:", tok[3])
print("Same token vector?", torch.equal(tok[2], tok[3]))

print("\n'l' FINAL embedding at position 2:", final[2])
print("'l' FINAL embedding at position 3:", final[3])
print("Same final vector?", torch.equal(final[2], final[3]))