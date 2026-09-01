"""
Class Index -> Attack Name Mapper

Recovers the mapping between encoded integer class labels and the
original CICIoT2023 attack names, then prints it alongside sample
counts, flagging which classes are currently held out as zero-day.

Tries three strategies, in order:

1. A saved label encoder artifact (label_encoder.pkl / classes.npy)
   anywhere under data/.
2. String labels still present in a processed label file.
3. Reconstruction from the raw CICIoT2023 label column: sklearn's
   LabelEncoder assigns indices by SORTED order of the unique label
   strings, so sorting the raw unique labels alphabetically
   reproduces index -> name exactly (provided LabelEncoder was used).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import ZERO_DAY_CLASSES


def try_saved_encoder():
    """Strategy 1: look for a saved encoder artifact."""

    candidates = list(Path("data").rglob("label_encoder*")) + \
                 list(Path("data").rglob("classes.npy")) + \
                 list(Path("data").rglob("label_mapping*"))

    for path in candidates:

        try:

            if path.suffix == ".npy":

                classes = np.load(path, allow_pickle=True)

                return list(classes), f"saved artifact: {path}"

            if path.suffix in (".pkl", ".joblib"):

                import joblib

                encoder = joblib.load(path)

                return list(encoder.classes_), f"saved artifact: {path}"

            if path.suffix == ".csv":

                df = pd.read_csv(path)

                # Assume two columns: index, name (order-agnostic)
                name_col = df.select_dtypes(include="object").columns[0]

                return list(df[name_col]), f"saved artifact: {path}"

        except Exception:

            continue

    return None, None


def try_processed_strings():
    """Strategy 2: label files that still contain strings."""

    for split in ["train", "validation", "test"]:

        for candidate in Path("data").rglob(f"y_{split}.csv"):

            try:

                y = pd.read_csv(candidate).iloc[:, 0]

                if y.dtype == object:

                    classes = sorted(y.unique())

                    return classes, f"string labels in: {candidate}"

            except Exception:

                continue

    return None, None


def try_raw_reconstruction():
    """
    Strategy 3: reconstruct from raw CICIoT2023 CSVs.

    sklearn LabelEncoder assigns 0..N-1 to the SORTED unique label
    strings, so sorting raw unique labels alphabetically reproduces
    the exact mapping — valid only if LabelEncoder (or equivalent
    sorted mapping) was used in preprocessing.
    """

    raw_dir = Path("data/raw/ciciot2023")

    if not raw_dir.exists():

        return None, None

    label_values = set()

    csv_files = list(raw_dir.rglob("*.csv"))

    if not csv_files:

        return None, None

    for csv_path in csv_files:

        try:

            # Only need the label column; find it cheaply from header
            header = pd.read_csv(csv_path, nrows=0)

            label_col = None

            for col in header.columns:

                if col.strip().lower() in ("label", "attack", "class", "attack_type"):

                    label_col = col

                    break

            if label_col is None:

                continue

            labels = pd.read_csv(csv_path, usecols=[label_col])[label_col]

            label_values.update(labels.unique())

        except Exception:

            continue

    if not label_values:

        return None, None

    return sorted(label_values), f"reconstructed from raw CSVs in {raw_dir} (sorted-unique, LabelEncoder convention)"


def main():

    classes, source = try_saved_encoder()

    if classes is None:

        classes, source = try_processed_strings()

    if classes is None:

        classes, source = try_raw_reconstruction()

    if classes is None:

        print(
            "Could not recover the class mapping automatically.\n"
            "None of the strategies found a source:\n"
            "  1. No saved encoder artifact under data/\n"
            "  2. No string labels in processed y_*.csv files\n"
            "  3. No raw label column found under data/raw/ciciot2023\n\n"
            "Check src/data/preprocess.py to see how labels were "
            "encoded, and where (if anywhere) the encoder was saved."
        )

        return

    print("=" * 70)
    print(f"Mapping source: {source}")
    print("=" * 70)

    # Sample counts from the encoded training labels

    counts = {}

    try:

        from src.data.dataset import CICIoTDataset

        train_dataset = CICIoTDataset(

            feature_file="X_train.csv",

            label_file="y_train.csv",

        )

        labels, cnts = np.unique(

            train_dataset.y.numpy(),

            return_counts=True,

        )

        counts = dict(zip(labels, cnts))

    except Exception as e:

        print(f"(Could not load train dataset for counts: {e})\n")

    zero_day_set = set(ZERO_DAY_CLASSES)

    print(f"\n{'Idx':>4}  {'Attack Name':<45} {'Train Samples':>13}  Zero-Day?")
    print("-" * 78)

    for idx, name in enumerate(classes):

        count = counts.get(idx, "?")

        flag = "  <-- HELD OUT" if idx in zero_day_set else ""

        print(f"{idx:>4}  {str(name):<45} {str(count):>13}{flag}")

    print("-" * 78)

    print(f"\nTotal classes : {len(classes)}")

    print(f"Zero-day held : {sorted(zero_day_set)}")

    # Sanity check

    missing = [c for c in zero_day_set if c >= len(classes)]

    if missing:

        print(
            f"\nWARNING: ZERO_DAY_CLASSES contains indices {missing} "
            f"that exceed the number of recovered classes "
            f"({len(classes)}). The mapping source may not match "
            f"your actual encoding — verify against preprocess.py."
        )


if __name__ == "__main__":

    main()