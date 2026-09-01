"""
Data Preprocessing Pipeline for CICIoT2023
------------------------------------------
This module loads the raw datasets, performs preprocessing,
encodes labels, scales features, and saves processed datasets.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ciciot2023"

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

train = pd.read_csv(RAW_PATH / "train" / "train.csv")
validation = pd.read_csv(RAW_PATH / "validation" / "validation.csv")
test = pd.read_csv(RAW_PATH / "test" / "test.csv")

print("Datasets loaded successfully.")


# -------------------------------------------------------------------
# Separate Features and Labels
# -------------------------------------------------------------------

X_train = train.drop(columns=["label"])
y_train = train["label"]

X_val = validation.drop(columns=["label"])
y_val = validation["label"]

X_test = test.drop(columns=["label"])
y_test = test["label"]


# -------------------------------------------------------------------
# Label Encoding
# -------------------------------------------------------------------

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)

y_val = label_encoder.transform(y_val)

y_test = label_encoder.transform(y_test)


# -------------------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


# -------------------------------------------------------------------
# Save processed datasets
# -------------------------------------------------------------------

pd.DataFrame(X_train).to_csv(PROCESSED_PATH / "X_train.csv", index=False)
pd.DataFrame(X_val).to_csv(PROCESSED_PATH / "X_validation.csv", index=False)
pd.DataFrame(X_test).to_csv(PROCESSED_PATH / "X_test.csv", index=False)

pd.DataFrame({"label": y_train}).to_csv(PROCESSED_PATH / "y_train.csv", index=False)
pd.DataFrame({"label": y_val}).to_csv(PROCESSED_PATH / "y_validation.csv", index=False)
pd.DataFrame({"label": y_test}).to_csv(PROCESSED_PATH / "y_test.csv", index=False)


# -------------------------------------------------------------------
# Save preprocessing objects
# -------------------------------------------------------------------

joblib.dump(scaler, PROCESSED_PATH / "scaler.pkl")
joblib.dump(label_encoder, PROCESSED_PATH / "label_encoder.pkl")


print("Preprocessing complete.")