import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
target = torch.tensor(10.0)

y = x * w + b
loss = (y - target) ** 2

print("=== Forward Pass ===")
print(f"y = {y.item()}")
print(f"loss = {loss.item()}")

loss.backward()

print("\n=== PyTorch Gradients ===")
print(f"dw = {w.grad.item()}")
print(f"db = {b.grad.item()}")

# Now let's compute by hand
print("\n=== Manual Gradients ===")
y_val = y.item()
dloss_dy = 2 * (y_val - target.item())  # derivative of (y-10)²
dy_dw = x.item()                         # derivative of x*w+b with respect to w
dy_db = 1.0                              # derivative of x*w+b with respect to b

dw_manual = dloss_dy * dy_dw
db_manual = dloss_dy * dy_db

print(f"dloss/dy = {dloss_dy}")
print(f"dy/dw    = {dy_dw}")
print(f"dw manual = {dw_manual}")
print(f"db manual = {db_manual}")

print("\n=== Match? ===")
print(f"dw: PyTorch={w.grad.item()} Manual={dw_manual} Match={w.grad.item()==dw_manual}")
print(f"db: PyTorch={b.grad.item()} Manual={db_manual} Match={b.grad.item()==db_manual}")