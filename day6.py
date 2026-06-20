import torch
import torch.nn as nn

x = torch.tensor([ [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]])

y = torch.tensor([[0.0],
    [1.0],
    [1.0],
    [0.0]])

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.layer1(x)
        x = self.sigmoid(x)
        x=self.layer2(x)
        x = self.sigmoid(x)
        return x

model = XORNet()

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=0.5)

print("=== Before Training ===")
print(model(x))

# Training loop
for epoch in range(5000):
    # Forward pass
    predictions = model(x)
    loss = loss_fn(predictions, y)

    # Backward pass
    optimizer.zero_grad()  # clear old gradients
    loss.backward()        # compute new gradients
    optimizer.step()       # update weights

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

print("\n=== After Training ===")
print(model(x))
print("\nTarget:")
print(y)