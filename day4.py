import torch

a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])

print("a:", a)
print("b:", b)
print("type:",a.dtype)

print("\na + b:", a + b)
print("a * b:", a * b)
print("dot product:", torch.dot(a, b))

matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

print("\nMatrix:\n", matrix)
print("Shape:", matrix.shape)

random_tensor = torch.randn(3, 3)
print("\nRandom tensor:\n", random_tensor)

result = torch.matmul(matrix, random_tensor)
print("\nMatrix multiply result:\n", result)
print("Shape:", result.shape)

import torch

# requires_grad=True tells PyTorch:
# "track every operation on this tensor"
# "I want to compute gradients for it later"
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# This is a neuron! (same as Day 2)
# y = x*w + b
y = x * w + b
print("y:", y)

# This is the "loss" — how wrong our neuron is
# Imagine the correct answer should be 10
# Our neuron gave us y, so the error is:
loss = (y - 10) ** 2
print("loss:", loss)

# NOW THE MAGIC — compute gradients
# This tells PyTorch: figure out how much
# each weight contributed to the loss
loss.backward()

# Gradients — how much should each weight change?
print("\nGradients:")
print("dw:", w.grad)  # how much to adjust w
print("db:", b.grad)  # how much to adjust b
print("dx:", x.grad)  # how much to adjust x