"""
Central configuration: paths, audio settings, feature layout, model hyperparameters.

Everything that was hardcoded to Google Drive in the notebook lives here instead,
so the pipeline can run locally or in Colab by changing one file.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"   # cached .pkl feature tables
MODELS_DIR = ARTIFACTS_DIR / "models"       # saved sklearn pipelines

# Where the raw Fake-or-Real audio lives. Override with the FOR_DATA_ROOT
# environment variable if you move the dataset.
#
# Expected layout under this root:
#   for-2sec/for-2seconds/training/{real,fake}/*.wav
#   for-2sec/for-2seconds/testing/{real,fake}/*.wav
#   for-rerec/for-rerecorded/validation/{real,fake}/*.wav
# Default: a "data" directory in the repo root. Point elsewhere either by
# editing DEFAULT_DATA_ROOT, by setting the FOR_DATA_ROOT environment
# variable, or by symlinking:  ln -s /path/to/your/dataset data
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT = Path(os.environ.get("FOR_DATA_ROOT", DEFAULT_DATA_ROOT))

SUBSET_PATHS = {
    "train": "for-2sec/for-2seconds/training",
    "test": "for-2sec/for-2seconds/testing",
    "rerec": "for-rerec/for-rerecorded/validation",
}

# --------------------------------------------------------------------------
# Audio / feature extraction
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000     # every file is resampled to this
N_MFCC = 13             # number of MFCC coefficients (and delta-MFCCs)
N_CHROMA = 12           # chroma_stft always returns 12 pitch classes

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".webm", ".m4a", ".ogg")

# Label convention, set by build_dataset() and used everywhere downstream.
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "Human", LABEL_FAKE: "AI-generated"}

# --------------------------------------------------------------------------
# Model hyperparameters (the winners from GridSearchCV in the notebook)
# --------------------------------------------------------------------------
SVM_PARAMS = {"kernel": "rbf", "C": 10, "gamma": 0.01}
LR_PARAMS = {"C": 0.1, "penalty": "l1", "solver": "liblinear", "max_iter": 500}
RF_PARAMS = {"n_estimators": 100, "random_state": 42}

# Search grids, used only when train.py is run with --tune
SVM_GRID = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001],
    "clf__kernel": ["rbf"],
}
LR_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear", "saga"],
    "clf__max_iter": [500, 1000],
}

# Weighted soft-vote ensembles -- the paper's "LORIS".
# The digit in the name IS the LR weight x 10: LORIS7 = 0.7 LR / 0.3 SVM.
# The paper never expands the acronym.
ENSEMBLE_WEIGHTS = {
    "loris7": {"svm": 0.3, "lr": 0.7},
    "loris9": {"svm": 0.1, "lr": 0.9},
}

# The weight sweep that PRODUCED those two numbers (Figure 4 in the paper).
# Only LR weight is swept; SVM weight is always 1 - LR weight, so there is
# exactly one knob. The range starts at 0.5 because SVM was already known to
# be class-biased -- it is never given the larger share.
ENSEMBLE_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

RANDOM_STATE = 42
CV_FOLDS = 5


def subset_dir(data_root, subset):
    """Resolve a subset name ('train'/'test'/'rerec') to its directory."""
    if subset not in SUBSET_PATHS:
        raise KeyError(f"Unknown subset {subset!r}. Choose from {list(SUBSET_PATHS)}")
    return Path(data_root) / SUBSET_PATHS[subset]
