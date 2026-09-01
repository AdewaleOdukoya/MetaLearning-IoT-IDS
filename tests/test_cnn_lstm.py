import torch

from src.models.cnn_lstm import CNNLSTM


model = CNNLSTM(

    input_dim=46,

    num_classes=34,

)

print(model)

x = torch.randn(32,46)

y = model(x)

print()

print("Input :", x.shape)

print("Output:", y.shape)