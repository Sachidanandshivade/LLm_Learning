text = "hello worlld"

chars = sorted(list(set(text)))
print("Unique characters:", chars)
print("Vocabulary size:", len(chars))

char_to_id = {ch: i for i,ch in enumerate(chars)}
print("\n Char to ID:", char_to_id)

id_to_char={i: ch for i,ch in enumerate(chars)}
print("\n ID to Char:", id_to_char)

def encode(text):
    return [char_to_id[ch] for ch in text]

def decode(ids):
    return "".join([id_to_char[i] for i in ids])

encoded = encode("hello")
print("\nEncoded 'hello':", encoded)

decoded = decode(encoded)
print("Decoded back:", decoded)