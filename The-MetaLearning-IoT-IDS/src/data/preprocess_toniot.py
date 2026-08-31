"""
Data Preprocessing Pipeline for TON-IoT (Train_Test_Network.csv)
----------------------------------------------------------------
Mirrors the CICIoT2023 pipeline (label encoding, standard scaling,
same output file naming) but handles TON-IoT's specifics:

  - Drops identifier / leakage columns (IPs, ports) and
    high-cardinality free-text columns (URIs, DNS queries,
    certificate subjects, user agents), per the proposal's
    preprocessing design.
  - One-hot encodes low-cardinality categorical columns
    ('-' placeholders treated as their own 'missing' category).
  - Coerces numeric-like columns (which contain '-') to numeric.
  - Uses the multiclass `type` column as the label (drops the
    binary `label` column).
  - Stratified 70/15/15 train/validation/test split.

Outputs to data/processed/toniot/ — completely separate from the
CICIoT2023 artifacts in data/processed/.

Usage:
    python -m src.data.preprocess_toniot
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "toniot" / "Train_Test_Network.csv"

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "toniot"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Column groups
# -------------------------------------------------------------------

# Identifier / leakage features (proposal: "Removal of identifier
# and leakage features (IPs, ports where appropriate, timestamps)")
DROP_IDENTIFIERS = [
    "src_ip", "src_port", "dst_ip", "dst_port",
]

# High-cardinality free-text — near-unique per flow, useless for
# generalisation and would explode one-hot dimensionality
DROP_HIGH_CARDINALITY = [
    "dns_query", "ssl_subject", "ssl_issuer",
    "http_uri", "http_user_agent", "weird_addl",
]

# Low-cardinality categoricals -> one-hot
CATEGORICAL = [
    "proto", "service", "conn_state",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "http_method", "http_version",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_notice",
]

# Numeric columns that may contain '-' placeholders
NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "http_trans_depth", "http_request_body_len",
    "http_response_body_len", "http_status_code",
]

LABEL_COLUMN = "type"

BINARY_LABEL_COLUMN = "label"   # dropped — we do multiclass


def main():

    print("Loading TON-IoT ...")

    df = pd.read_csv(RAW_FILE, low_memory=False)

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    print("\nClass distribution (type):")
    print(df[LABEL_COLUMN].value_counts())

    # ---------------------------------------------------------------
    # Drop identifiers, free-text, and the binary label
    # ---------------------------------------------------------------

    df = df.drop(
        columns=DROP_IDENTIFIERS
        + DROP_HIGH_CARDINALITY
        + [BINARY_LABEL_COLUMN]
    )

    # ---------------------------------------------------------------
    # Numeric coercion ('-' -> NaN -> 0)
    # ---------------------------------------------------------------

    for col in NUMERIC:

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---------------------------------------------------------------
    # One-hot encode categoricals ('-' becomes its own category,
    # which is meaningful: "this protocol field was absent")
    # ---------------------------------------------------------------

    df[CATEGORICAL] = df[CATEGORICAL].astype(str)

    df = pd.get_dummies(
        df, columns=CATEGORICAL, dtype=np.float32
    )

    print(f"\nFeature count after one-hot encoding: {df.shape[1] - 1}")

    # ---------------------------------------------------------------
    # Features / labels
    # ---------------------------------------------------------------

    y = df[LABEL_COLUMN]

    X = df.drop(columns=[LABEL_COLUMN])

    feature_names = list(X.columns)

    # ---------------------------------------------------------------
    # Label encoding (LabelEncoder = sorted alphabetical order,
    # same convention as the CICIoT2023 pipeline)
    # ---------------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print("\nLabel mapping:")

    for idx, name in enumerate(label_encoder.classes_):

        count = int((y_encoded == idx).sum())

        print(f"  {idx}: {name}  ({count:,} samples)")

    # ---------------------------------------------------------------
    # Stratified 70 / 15 / 15 split
    # ---------------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values, y_encoded,
        test_size=0.30,
        stratify=y_encoded,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    print(f"\nSplit sizes: train={len(y_train):,} "
          f"val={len(y_val):,} test={len(y_test):,}")

    # ---------------------------------------------------------------
    # Scaling (fit on train only)
    # ---------------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------------
    # Save (same filename pattern as CICIoT2023)
    # ---------------------------------------------------------------

    pd.DataFrame(X_train).to_csv(PROCESSED_PATH / "X_train.csv", index=False)
    pd.DataFrame(X_val).to_csv(PROCESSED_PATH / "X_validation.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_PATH / "X_test.csv", index=False)

    pd.DataFrame({"label": y_train}).to_csv(
        PROCESSED_PATH / "y_train.csv", index=False)
    pd.DataFrame({"label": y_val}).to_csv(
        PROCESSED_PATH / "y_validation.csv", index=False)
    pd.DataFrame({"label": y_test}).to_csv(
        PROCESSED_PATH / "y_test.csv", index=False)

    joblib.dump(scaler, PROCESSED_PATH / "scaler.pkl")
    joblib.dump(label_encoder, PROCESSED_PATH / "label_encoder.pkl")

    with open(PROCESSED_PATH / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    print("\nTON-IoT preprocessing complete.")
    print(f"Artifacts written to {PROCESSED_PATH}")


if __name__ == "__main__":

    main()