import torch

x = torch.tensor([1.0,2.0,3.0]);

w1 = torch.randn(3,2,requires_grad=True)
b1 = torch.randn(2, requires_grad=True)

w2 = torch.randn(2,1,requires_grad=True)
b2 = torch.randn(1,requires_grad=True)

layer1_out = torch.matmul(x,w1)+b1
print(layer1_out)

layer2_out = torch.matmul(layer1_out,w2)+b2
print(layer2_out)

target = torch.tensor([10.0])
loss = (layer2_out-target)**2
print(loss)
loss = loss.mean()
print(loss)
loss.backward()

print("\nGradients computed for:")
print("W1 grad shape:", w1.grad.shape)
print("W2 grad shape:", w2.grad.shape)
print("PyTorch handled backprop through 2 layers automatically!")