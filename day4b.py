import torch

# Reset weights
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
target = torch.tensor(10.0)  # correct answer
lr = 0.01

print("\n--- Training Loop ---")
for step in range(1000):
    # Forward pass — compute prediction
    y = x * w + b

    # Compute loss
    loss = (y - target) ** 2

    # Backward pass — compute gradients
    loss.backward()

    # Update weights manually
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Zero gradients — IMPORTANT! otherwise they accumulate
    w.grad.zero_()
    b.grad.zero_()

    # Print every 100 steps
    if step % 100 == 0:
        print(f"Step {step}: y={y.item():.4f} loss={loss.item():.4f}")

print(f"\nFinal y: {y.item():.4f}")
print(f"Target:  {target.item():.4f}")