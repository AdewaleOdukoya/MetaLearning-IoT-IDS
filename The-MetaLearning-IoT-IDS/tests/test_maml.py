import torch

from src.models.maml import MAML

model = MAML(

    input_dim=46,

    num_classes=5,

)

x = torch.randn(25, 46)

y = model(x)

print(model)

print()

print("Output Shape:", y.shape)

print()

print("MAML Model Test Passed.")