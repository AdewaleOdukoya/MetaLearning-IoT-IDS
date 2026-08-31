import torch

from src.models.mlp import MLP


INPUT_DIM = 46
NUM_CLASSES = 34

model = MLP(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES
)

print(model)

dummy = torch.randn(32, INPUT_DIM)

output = model(dummy)

print()

print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)