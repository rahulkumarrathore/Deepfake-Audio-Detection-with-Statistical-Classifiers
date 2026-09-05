"""
STAGE 1 -- DATA LOADING & PREPROCESSING   (direct: no cache, runs every time)
============================================================================

    audio file
        -> load_audio()        16 kHz signal
        -> extract_features()  84 numbers
        -> one dict per clip in dict_audio_train / dict_audio_test
        -> DataFrame  ->  X (84 columns), y (0/1)

The dicts keep the per-clip bookkeeping (subject_id, duration, class name);
the DataFrames are what the classifiers actually consume.

Importing this module BUILDS the train and test sets immediately. That is the
point -- there is no .pkl cache and nothing is reused between runs. The
rerecorded split is NOT built on import (it is only needed for the benchmark);
call build_bucket(list_wav_paths_rerec) when you want it.

Run directly to see the tables:
    python -m src.data_preprocessing
"""

import os

import pandas as pd

from src import config
from src.dataset_directory import *          # noqa: F403  -- the path lists
from src.features import extract_features, feature_names, load_audio

USE_CHROMA = True          # False -> the reduced 60-D feature set

dict_audio_train = []
dict_audio_test = []


# ==========================================================================
# 1. Label -- comes from the folder name, nowhere else
# ==========================================================================
def get_class_from_path(path_wav):
    """real -> 0, fake -> 1. This is where the label convention is set."""
    folder_name = os.path.basename(os.path.dirname(path_wav)).lower()
    if "real" in folder_name:
        return config.LABEL_REAL, "real"
    elif "fake" in folder_name:
        return config.LABEL_FAKE, "fake"
    else:
        raise ValueError(f"Unknown folder type {folder_name}")


# ==========================================================================
# 2. One file -> one dict
# ==========================================================================
def get_structured_dict_files(path_wav, bucket):
    """Load one clip, extract its features, append the dict to `bucket`."""
    signal, sr = load_audio(path_wav)
    second = len(signal) / sr
    features = extract_features(signal, sr, USE_CHROMA)
    class_no, class_name = get_class_from_path(path_wav)
    subject_id = os.path.basename(path_wav).split(".")[0]
    temp = {"features": features,
            "sr": sr,
            "second": second,
            "class": class_no,
            "class_name": class_name,
            "subject_id": subject_id,
            "filename": path_wav}
    bucket.append(temp)
    return temp


def build_bucket(paths, name=""):
    """Run get_structured_dict_files over a list of paths. Corrupt files are
    skipped so one bad clip cannot kill the whole run."""
    bucket = []
    for n, path in enumerate(paths, 1):
        try:
            get_structured_dict_files(path, bucket)
        except Exception as exc:
            print(f"    ! skipped {path}: {exc}")
        if n % 500 == 0:
            print(f"    ... {n}/{len(paths)}")
    if name:
        print(f"length of dict_audio_{name}: {len(bucket)}")
    return bucket


# ==========================================================================
# 3. dicts -> DataFrame -> X, y
# ==========================================================================
def to_dataframe(dicts, use_chroma=USE_CHROMA):
    """One row per clip: filename + 84 features + label."""
    rows = [[d["filename"]] + d["features"] + [d["class"]] for d in dicts]
    columns = ["filename"] + feature_names(use_chroma) + ["label"]
    return pd.DataFrame(rows, columns=columns)


def split_xy(df):
    """
    Separate features from the label.

    `filename` is dropped because it is bookkeeping, not signal -- leaving it
    in would let the model learn from file naming rather than from audio.
    """
    X = df.drop(columns=["filename", "label"])
    y = df["label"]
    return X, y


def drop_chroma(X):
    """Remove the 12 chroma pairs (84-D -> 60-D) for the reduction experiment."""
    return X.loc[:, ~X.columns.str.startswith("chroma")]


# NOTE: there is deliberately NO StandardScaler here.
# Scaling lives INSIDE the sklearn Pipeline built in train.py, so it is fitted
# on training data only and travels with the saved model.


# ==========================================================================
# 4. Build it -- this runs on import
# ==========================================================================
print("######### extracting features ###########")
dict_audio_train = build_bucket(list_wav_paths_train, "train")   # noqa: F405
dict_audio_test = build_bucket(list_wav_paths_test, "test")      # noqa: F405

train_df = to_dataframe(dict_audio_train)
test_df = to_dataframe(dict_audio_test)

X_train, y_train = split_xy(train_df)
X_test, y_test = split_xy(test_df)

print("######### data structured ###########")
print("train:", X_train.shape, "| test:", X_test.shape)


if __name__ == "__main__":
    print("\ntrain label counts:")
    print(y_train.value_counts().rename(config.LABEL_NAMES))
    print("\ntest label counts:")
    print(y_test.value_counts().rename(config.LABEL_NAMES))
    print("\nfirst 5 columns of the first 3 training rows:")
    print(X_train.iloc[:3, :5])
