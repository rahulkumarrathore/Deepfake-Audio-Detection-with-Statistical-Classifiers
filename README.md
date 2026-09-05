# Deepfake Audio Detection with Statistical Classifiers

Detecting AI-generated speech with classical machine learning — 84 handcrafted acoustic features, no neural networks.

**The finding:** the SVM gets **93.9% of real clips right and 33.3% of fake ones**. Blend those and you get a respectable-looking 64% that hides the failure completely. Logistic Regression gives up 22 points on real audio to reach **56.6% on fake**, and is the only model that stays balanced across both classes.

---

## Results

Three sets are used. Nothing is tuned on the last two — they are scored once, at the end.

| Directory | Clips | Used as |
|---|---|---|
| `for-2sec/.../training` | 13,956 | training |
| `for-2sec/.../testing` | 1,088 | held-out test |
| `for-rerec/.../validation` | 2,244 | second, harder test (out of distribution) |

The third is named `validation` by the dataset, but nothing is tuned on it, so it acts as a test set.

### Held-out `testing` split — the key result

1,088 clips, balanced. Real and fake are scored **separately**.

| Model | Accuracy | Weighted F1 | Real | Fake |
|---|---|---|---|---|
| Random Forest | 0.4982 | 0.4918 | 0.6103 | 0.3860 |
| SVM | 0.6360 | 0.5992 | 0.9393 | 0.3327 |
| LR | 0.6443 | 0.6421 | 0.7224 | 0.5662 |
| LORIS7 | 0.6553 | 0.6448 | 0.8272 | 0.4835 |
| LORIS9 | 0.6452 | 0.6410 | 0.7537 | 0.5368 |

The SVM has the highest real-audio accuracy of any model and the second-lowest on fake audio. It has learned to answer "human" — free accuracy on real clips, near-total failure on the class the system exists to catch. Random Forest sits at chance level.

5-fold CV on the training split gives SVM 0.9898 and LR 0.9099 — against 0.64 for both on this test split. That ~35-point gap is far larger than ordinary generalization loss and suggests the testing split already differs in distribution from training.

### Ensemble weight sweep

Soft voting is a weighted average of the two models' probabilities, so there is one knob:

```
P_final = w_LR × P_LR  +  (1 − w_LR) × P_SVM
```

| w_LR | Real | Fake | Macro F1 | Gap |
|---|---|---|---|---|
| 0.50 | 0.9081 | 0.4007 | 0.6306 | 0.5074 |
| 0.60 | 0.8640 | 0.4449 | 0.6385 | 0.4191 |
| 0.70 | 0.8272 | 0.4835 | 0.6448 | 0.3437 |
| 0.80 | 0.7923 | 0.5092 | 0.6436 | 0.2831 |
| 0.90 | 0.7537 | 0.5368 | 0.6410 | **0.2169** |

Every step toward LR trades a little real accuracy for more fake accuracy, and the gap collapses from 0.51 to 0.22. The weight is not tuning *how accurate* the ensemble is — it is tuning **how much of the SVM's bias survives**.

Macro F1 stays flat across the whole range, so no weight is clearly best. The trend simply points at its own endpoint, `w_LR = 1.00` — plain LR, no ensemble.

### Takeaways

1. **Logistic Regression is the model to ship.** It has the smallest gap between its two class accuracies (72.2% real, 56.6% fake) and the highest weighted F1 of any single model.
2. **The SVM is biased toward "real".** It misses two out of three deepfakes. A detector with that failure mode has little practical value regardless of its headline score.
3. **The ensemble does not beat LR.** Both LORIS variants inherit some of the SVM's bias without gaining anything net.
4. **Per-class evaluation is not optional.** Overall accuracy barely moves across the entire sweep (0.641 to 0.645 macro F1) while the model's behaviour changes completely underneath it — real accuracy falls 15 points and fake accuracy rises 14.

---

## Installation

Requires **Python 3.10+** (tested on 3.13).

```bash
git clone https://github.com/rahulkumarrathore/Deepfake-Audio-Detection-with-Statistical-Classifiers.git
cd Deepfake-Audio-Detection-with-Statistical-Classifiers

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `scikit-learn<1.10` on purpose: the LR configuration uses `penalty='l1'` (removed in 1.10) and the ensembles use `SVC(probability=True)` (removed in 1.11).

In PyCharm, set the interpreter to `.venv/bin/python`.

---

## Dataset

Download the **[Fake-or-Real (FoR) dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)** from Kaggle (~19 GB) and unzip it. Only these directories are read (~1.2 GB):

```
<data root>/
├── for-2sec/for-2seconds/{training,testing}/{real,fake}/*.wav
└── for-rerec/for-rerecorded/validation/{real,fake}/*.wav
```

Point the project at it in any one of three ways:

```bash
export FOR_DATA_ROOT=/path/to/dataset   # 1. environment variable
ln -s /path/to/dataset data             # 2. symlink
                                        # 3. edit DEFAULT_DATA_ROOT in src/config.py
```

**Label convention:** the folder name *is* the label — `real → 0`, `fake → 1`. Set in one place, `get_class_from_path()`.

---

## Running

There are **no command-line arguments**. Each file has a `SETTINGS` block at the top — edit it and run.

```bash
python -m src.data_preprocessing   # Stage 1: audio -> 84-D features   (~66 s)
python -m src.train                # Stage 2: train + evaluate + sweep (~110 s)
python -m src.inference            # Stage 3: OOD benchmark            (~12 s)
```

Timings are for 15,044 clips on an Apple M4. No GPU needed; peak memory under 1 GB.

**`src/train.py` settings**

```python
MODELS = ["rf", "svm", "lr", "ensemble"]
USE_CHROMA = True     # False -> drop chroma (84-D -> 60-D), saves as *_nochroma
DO_CV = True          # 5-fold cross-validation
DO_SWEEP = True       # sweep the LR ensemble weight
DO_TUNE = False       # re-run GridSearchCV (slow)
SAVE = True
```

**`src/inference.py` settings**

```python
MODELS = ["rf", "svm", "lr", "loris7", "loris9"]
MODE = "rerec"        # "rerec" | "file" | "dir"
PREDICT_FILE = ""
PREDICT_DIR = ""
```

Stage 2 writes `artifacts/models/*.joblib`. Each saved model is a complete `Pipeline` with its own fitted scaler, so it predicts on new audio with no other setup.

There is **no feature cache** — extraction runs every time, so every number on screen comes from the code currently on disk.

---

## How It Works

### Features

Every clip becomes a fixed-length **84-D vector**, regardless of duration. Audio is loaded at 16 kHz, then seven families are computed with `librosa`:

| Family | Coefficients |
|---|---|
| MFCCs | 13 |
| Δ MFCCs | 13 |
| Chroma (STFT) | 12 |
| Spectral centroid / bandwidth | 1 + 1 |
| Zero-crossing rate | 1 |
| RMS energy | 1 |
| **Total** | **42** |

Each returns a `(coefficients × frames)` matrix whose width depends on clip length — so an SVM cannot consume it. Taking the **mean and standard deviation along the time axis** collapses that dimension:

```python
for family in families:
    features.extend(np.mean(family, axis=1))
    features.extend(np.std(family, axis=1))
```

**42 series × 2 statistics = 84 features.** The cost is that all temporal structure is discarded.

### Pipeline

```
 for-2sec/training  ─┐
 for-2sec/testing   ─┤  load_audio() → extract_features() → 84-D rows → DataFrame
 for-rerec/validation┘
                              │
              Pipeline([StandardScaler, classifier])
                              │
     ┌────────────┬───────────┴────────┬──────────────────┐
     RF          SVM                   LR          VotingClassifier
                                                   (soft, weighted)
                              │                          │
                     artifacts/models/*.joblib      loris7 / loris9
                              │
                  for-rerec/validation → per-class accuracy
```

The scaler lives **inside** the Pipeline, so a saved model is self-contained and cross-validation re-fits it per fold instead of leaking validation statistics into scaling.

### Models

| Model | Configuration |
|---|---|
| Random Forest | `n_estimators=100` — baseline, dropped (chance level) |
| **SVM (RBF)** | `C=10, gamma=0.01` — `GridSearchCV`, `cv=5`, `scoring='f1'` |
| **Logistic Regression** | `C=0.1, penalty='l1', solver='liblinear', max_iter=500` — same search |
| LORIS7 / LORIS9 | Soft-vote ensembles, 0.3/0.7 and 0.1/0.9 SVM/LR |

The digit in LORIS7 / LORIS9 **is** the LR weight × 10.

---

## Repository Layout

```
src/
├── config.py               paths, hyperparameters, label constants
├── dataset_directory.py    file-path lists (glob + natsort); no audio read
├── features.py             load_audio() + extract_features(); no module-level work
├── data_preprocessing.py   STAGE 1: audio -> 84-D table (builds on import)
├── train.py                STAGE 2: trained Pipelines + weight sweep
└── inference.py            STAGE 3: prediction + OOD benchmark
```

`data_preprocessing.py` builds the dataset on import, which is why inference imports `features.py` instead — otherwise classifying one clip would extract 15,000 files.


## Reference

Dataset: [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) (Kaggle)
