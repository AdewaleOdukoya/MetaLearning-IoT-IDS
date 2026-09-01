"""
Reproducibility Utilities
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic cuDNN (slower, but required for
    # reproducible results reported in the dissertation)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False