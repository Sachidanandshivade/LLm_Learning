from collections import Counter

from sympy import python

# Our training text
text = "low lower lowest low lower newest newest"

# Step 1 — start with character-level tokens
# We add </w> to mark end of word (so "low" and "lower" aren't confused)
words = text.split()
tokens = [" ".join(list(word)) + " </w>" for word in words]
print("Initial tokens:")
for t in tokens:
    print(" ", t)

# Step 2 — count all adjacent pairs
def get_pair_counts(tokens):
    pairs = Counter()
    for token in tokens:
        symbols = token.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += 1
    return pairs

pairs = get_pair_counts(tokens)
print("\nPair counts:")
for pair, count in pairs.most_common(5):
    print(" ", pair, ":", count)


# Step 3 — merge the most frequent pair
def merge_pair(tokens, pair):
    new_tokens = []
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for token in tokens:
        new_token = token.replace(bigram, replacement)
        new_tokens.append(new_token)
    return new_tokens
best_pair = pairs.most_common(1)[0][0]
print("\nMerging:", best_pair)
tokens = merge_pair(tokens, best_pair)
print("\nTokens after 1 merge:")
for t in tokens:
    print(" ", t)

print("\n\n=== Running Multiple Merges ===")

num_merges = 10
for i in range(num_merges):
    pairs = get_pair_counts(tokens)
    if not pairs:
        break
    best_pair = pairs.most_common(1)[0][0]
    tokens = merge_pair(tokens, best_pair)
    print(f"Merge {i+1}: {best_pair} -> {''.join(best_pair)}")

print("\nFinal tokens:")
for t in tokens:
    print(" ", t)