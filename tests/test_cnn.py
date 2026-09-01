import torch

from src.models.cnn import CNN

model = CNN(
    input_dim=46,
    num_classes=34,
)

print(model)

x = torch.randn(32,46)

output = model(x)

print()

print("Input :", x.shape)

print("Output:", output.shape)