import numpy as np

word_cat = np.array([0.2,0.9,0.1])
word_dog = np.array([0.3,0.8,0.2])
word_car = np.array([0.9,0.1,0.8])


# This is called "vectorized operations"
# LLMs do this on millions of numbers simultaneously
print("cat vs dog:", np.dot(word_cat ,word_dog))
print("cat vs car:", np.dot(word_cat , word_car))

print("\ncat + dog:", word_cat + word_dog)
print("cat * 2:  ", word_cat * 2)

# Matrix — multiple words at once
# Each row = one word as a vector
sentence = np.array([
    [0.2, 0.9, 0.1],  # "cat"
    [0.3, 0.8, 0.2],  # "sat"
    [0.9, 0.1, 0.8],  # "on"
    [0.1, 0.4, 0.6],  # "the"
    [0.7, 0.2, 0.5],  # "mat"
])

print("Shape:", sentence.shape)  # (5 rows, 3 columns)
print("First word vector:", sentence[0])
print("Third word vector:", sentence[2])

# Matrix multiplication — the CORE operation of every LLM
# This is how attention works
weights = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
])

result = np.matmul(sentence, weights)
print("\nAfter matrix multiply:")
print(result)
print("Result shape:", result.shape)

random_weights = np.random.randn(3,3)
print("\nRandom weights:")
print(random_weights)

numbers = np.array([1,2,3,4,5,6,7,8,9,10])
print("\nMean:",np.mean(numbers))
print("Std:",np.std(numbers)) #standard_deviationn

a = np.array([1,2,3,4,5,6])
print("\nOriginal:" , a.shape)
reshaped = a.reshape(2,3)
print("\nReshaped:" , reshaped)
print("New shape:" , reshaped.shape)
