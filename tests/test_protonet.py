import torch

from src.models.protonet import ProtoNet


model = ProtoNet(

    input_dim=46,

    embedding_dim=128,

)

print(model)

x = torch.randn(16,46)

embedding = model(x)

print()

print("Input :", x.shape)

print("Embedding:", embedding.shape)