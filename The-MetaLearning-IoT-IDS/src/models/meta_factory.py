"""
Meta Model Factory

Creates meta-learning models.

Supported Models

- ProtoNet
- MAML
- Reptile
"""

from src.models.protonet import ProtoNet
from src.models.maml import MAML
from src.config.config import N_WAY
from src.models.reptile import Reptile


def get_meta_model(
    model_name,
    input_dim,
):

    model_name = model_name.upper()

    if model_name == "PROTONET":

        return ProtoNet(
            input_dim=input_dim,
            embedding_dim=128,
        )

    # ---------------------------------------------------------

    elif model_name == "MAML":

        return MAML(

            input_dim=input_dim,

            num_classes=N_WAY,

        )

    # ---------------------------------------------------------

    elif model_name == "REPTILE":

        return Reptile(
            input_dim=input_dim,
            num_classes=N_WAY,
        )

    # ---------------------------------------------------------



    else:

        raise ValueError(
            f"Unknown Meta Model: {model_name}"
        )