"""
Model Factory

Creates baseline models based on the selected model name.
"""

from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.cnn_lstm import CNNLSTM


def get_model(
    model_name,
    input_dim,
    num_classes,
):

    model_name = model_name.upper()

    if model_name == "MLP":

        return MLP(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    elif model_name == "CNN":

        return CNN(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    
    elif model_name == "CNN_LSTM":

        return CNNLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )