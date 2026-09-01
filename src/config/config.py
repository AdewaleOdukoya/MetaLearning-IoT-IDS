import torch

# ============================================
# Training Mode
# ============================================
DATASET = "CICIOT"   #  "TONIOT" or "CICIOT"

MODEL_TYPE = "BASELINE" #META

MODEL_NAME = "CNN_LSTM" #MLP, CNN_LSTM, PROTONET, MAML, REPTILE
# =====================================
# Data
# =====================================

BATCH_SIZE = 2048

# =====================================
# Training
# =====================================

EPOCHS = 20

LEARNING_RATE = 1e-3

# =====================================
# Device
# =====================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Few-Shot
# =====================================

N_WAY = 5
K_SHOT = 5
QUERY_SIZE = 15
EPISODES = 1000

# =====================================
# Reproducibility
# =====================================

SEED = 42

# MAML HYPERPARAMETERS

INNER_LR = 0.01

INNER_STEPS = 5

META_LR = LEARNING_RATE

# =====================================
# REPTILE HYPERPARAMETERS
# =====================================

REPTILE_INNER_LR = 0.01

REPTILE_INNER_STEPS = 5

REPTILE_META_LR = 0.001


# Zero-Day Class Split
# =====================================

# Class labels (as encoded integers, matching train_dataset.y)
# to hold out ENTIRELY from meta-training. These are only ever
# seen at final zero-day evaluation time via the test set.
#
# Choose these BEFORE running any real experiments, and keep them
# fixed across all model comparisons (ProtoNet/MAML/Reptile) so
# the comparison is fair — every model must be evaluated against
# the exact same unseen classes.

ZERO_DAY_CLASSES = [23, 24, 25, 11, 18, 16, 27, 17] #CICIOT DATASET HIDDEN CLASSES

ZERO_DAY_EPISODES = 200

# =====================================
# Dataset Selection
# =====================================

TONIOT_ZERO_DAY_CLASSES = [1, 3, 6, 7, 9] #TONIOT HIDDEN CLASSES