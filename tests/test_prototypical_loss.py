import torch

from src.models.protonet import ProtoNet

from src.losses.prototypical_loss import PrototypicalLoss


model = ProtoNet(
    input_dim=46,
)

loss_fn = PrototypicalLoss()

# --------------------------------------------

support_x = torch.randn(25,46)

support_y = torch.tensor(

    [0]*5 +

    [1]*5 +

    [2]*5 +

    [3]*5 +

    [4]*5

)

query_x = torch.randn(50,46)

query_y = torch.tensor(

    [0]*10 +

    [1]*10 +

    [2]*10 +

    [3]*10 +

    [4]*10

)

support_embeddings = model(
    support_x
)

query_embeddings = model(
    query_x
)

result = loss_fn(

    support_embeddings,

    support_y,

    query_embeddings,

    query_y,

)

print()

print("Loss")

print(result["loss"])

print()

print("Accuracy")

print(result["accuracy"])

print()

print("Prototype Shape")

print(result["prototypes"].shape)

print()

print("Logits Shape")

print(result["logits"].shape)