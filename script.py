import numpy as np

# Two matrices
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication — the core of every LLM operation
result = np.matmul(a, b)

print("Matrix A:")
print(a)

print("\nMatrix B:")
print(b)

print("\nA x B =")
print(result)