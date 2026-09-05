"""
STAGE 3 -- INFERENCE
====================

Loads a saved Pipeline from Stage 2 and predicts on new audio.

    new audio file
        -> extract_features()        <- the SAME function used in training
        -> pipeline.predict()        <- scaler is inside the pipeline
        -> 0 = Human, 1 = AI-generated

Three entry points:

    predict_file(path)              one file
    predict_directory(dir)          a folder of files, extracted once
    evaluate_rerecorded(root)       the out-of-distribution benchmark,
                                    scored separately for real and fake

No command-line arguments -- edit the SETTINGS block below and hit Run.

    python -m src.inference

A note on efficiency
--------------------
The notebook ran 12 separate loops over the rerecorded set (6 models x 2
classes), re-extracting features from all ~2,244 files every time -- roughly
27,000 extractions where 2,244 would do. Here, features are extracted ONCE
into a matrix and every model predicts over that same matrix, which makes the
benchmark about 12x faster.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import extract_features, feature_names, load_audio


# ==========================================================================
# SETTINGS -- change these, then just Run. No command-line flags.
# ==========================================================================
MODELS = ["rf", "svm", "lr", "loris7", "loris9"]

MODE = "rerec"        # "rerec" -> the out-of-distribution benchmark (Fig 3/5)
                      # "file"  -> predict PREDICT_FILE
                      # "dir"   -> predict every audio file in PREDICT_DIR

PREDICT_FILE = ""     # used when MODE == "file"
PREDICT_DIR = ""      # used when MODE == "dir"

USE_CHROMA = True     # False -> load the "_nochroma" 60-D models


# ==========================================================================
# 1. Loading a saved model
# ==========================================================================
def load_model(name: str, use_chroma: bool = True):
    """
    Load a trained Pipeline by name ('svm', 'lr', 'loris1', 'loris2', ...).

    The returned object contains BOTH the fitted StandardScaler and the
    classifier, so it can predict without any other setup. In the notebook
    this was not true: the scaler was never saved, so a loaded model was
    unusable until you rebuilt the scaler from the training data.
    """
    suffix = "" if use_chroma else "_nochroma"
    path = config.MODELS_DIR / f"{name}{suffix}.joblib"
    if not path.exists():
        available = sorted(p.stem for p in config.MODELS_DIR.glob("*.joblib"))
        raise FileNotFoundError(
            f"No model at {path}\n"
            f"Available: {available or '(none -- run: python -m src.train)'}"
        )
    return joblib.load(path)


def _expects_chroma(pipe) -> bool:
    """Infer the feature set a pipeline was trained on, from its input width."""
    n = getattr(pipe.named_steps["scaler"], "n_features_in_", None)
    return n != 60


# ==========================================================================
# 2. Feature extraction for inference
# ==========================================================================
def features_frame(paths, use_chroma: bool = True) -> pd.DataFrame:
    """
    Extract features for a list of files into one DataFrame.

    Column names are attached so the DataFrame matches what the pipeline saw
    during training. Passing a bare list would still work but triggers
    sklearn's "X does not have valid feature names" warning -- the notebook
    suppressed that warning; here we simply avoid causing it.
    """
    rows, kept = [], []
    for p in paths:
        try:
            signal, sr = load_audio(p)
            rows.append(extract_features(signal, sr, use_chroma))
            kept.append(str(p))
        except Exception as exc:
            print(f"  ! skipped {p}: {exc}")
    if not rows:
        raise RuntimeError("No files could be processed.")
    return pd.DataFrame(rows, columns=feature_names(use_chroma), index=kept)


def list_audio(directory) -> list[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in config.AUDIO_EXTENSIONS)


# ==========================================================================
# 3. Prediction
# ==========================================================================
def predict_file(pipe, file_path) -> dict:
    """Predict a single audio file. Returns label, name, and confidence."""
    X = features_frame([file_path], _expects_chroma(pipe))
    label = int(pipe.predict(X)[0])
    out = {"file": str(file_path), "label": label, "prediction": config.LABEL_NAMES[label]}
    if hasattr(pipe, "predict_proba"):
        out["confidence"] = float(pipe.predict_proba(X)[0][label])
    return out


def predict_directory(pipe, directory) -> pd.DataFrame:
    """Predict every audio file in a directory. Features extracted once."""
    paths = list_audio(directory)
    if not paths:
        raise RuntimeError(f"No audio files found in {directory}")
    X = features_frame(paths, _expects_chroma(pipe))
    preds = pipe.predict(X)

    df = pd.DataFrame({
        "file": [Path(p).name for p in X.index],
        "label": preds,
        "prediction": [config.LABEL_NAMES[int(p)] for p in preds],
    })
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        df["confidence"] = [proba[i][int(p)] for i, p in enumerate(preds)]
    return df


# ==========================================================================
# 4. The out-of-distribution benchmark
# ==========================================================================
def evaluate_rerecorded(model_names, data_root, use_chroma: bool = True) -> pd.DataFrame:
    """
    Score models on the rerecorded validation set, PER CLASS.

    This is the experiment that carries the whole project. Real and fake are
    scored in separate columns on purpose: SVM reaches ~89% on real audio and
    ~33% on fake, and a single blended accuracy would average those into a
    respectable-looking ~61% and hide the failure completely.
    """
    rerec = config.subset_dir(data_root, "rerec")
    real_files = list_audio(rerec / "real")
    fake_files = list_audio(rerec / "fake")
    print(f"real: {len(real_files)} files\nfake: {len(fake_files)} files")

    # --- extract ONCE, reuse for every model ---
    print("\nExtracting features (once for all models) ...")
    X_real = features_frame(real_files, use_chroma)
    X_fake = features_frame(fake_files, use_chroma)

    rows = []
    for name in model_names:
        pipe = load_model(name, use_chroma)
        # correct on real = predicted 0;  correct on fake = predicted 1
        real_acc = float((pipe.predict(X_real) == config.LABEL_REAL).mean())
        fake_acc = float((pipe.predict(X_fake) == config.LABEL_FAKE).mean())
        rows.append({
            "model": name,
            "real_accuracy": real_acc,
            "fake_accuracy": fake_acc,
            "balanced_accuracy": (real_acc + fake_acc) / 2,
            "gap": abs(real_acc - fake_acc),
        })
        print(f"  {name:<10} real {real_acc:6.2%}   fake {fake_acc:6.2%}   "
              f"gap {abs(real_acc - fake_acc):5.2%}")

    results = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    print("\n" + "=" * 70)
    print("RERECORDED BENCHMARK (out-of-distribution)")
    print("=" * 70)
    print(results.to_string(index=False,
                            formatters={c: "{:.2%}".format for c in
                                        ["real_accuracy", "fake_accuracy",
                                         "balanced_accuracy", "gap"]}))
    print("\nA large gap means the model is biased toward one class.")
    return results


# ==========================================================================
# CLI
# ==========================================================================
if __name__ == "__main__":
    if MODE == "rerec":
        evaluate_rerecorded(MODELS, config.DATA_ROOT, USE_CHROMA)

    elif MODE == "file":
        if not PREDICT_FILE:
            raise SystemExit("Set PREDICT_FILE in the SETTINGS block above.")
        for name in MODELS:
            r = predict_file(load_model(name, USE_CHROMA), PREDICT_FILE)
            conf = f"  (confidence {r['confidence']:.2%})" if "confidence" in r else ""
            print(f"[{name}] {Path(r['file']).name}: {r['prediction']}{conf}")

    elif MODE == "dir":
        if not PREDICT_DIR:
            raise SystemExit("Set PREDICT_DIR in the SETTINGS block above.")
        for name in MODELS:
            print(f"\n=== {name} ===")
            print(predict_directory(load_model(name, USE_CHROMA), PREDICT_DIR)
                  .to_string(index=False))

    else:
        raise SystemExit(f"Unknown MODE {MODE!r}: choose 'rerec', 'file' or 'dir'.")
