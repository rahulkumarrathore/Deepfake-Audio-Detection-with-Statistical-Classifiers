# Deepfake Audio Detection with Statistical Classifiers

Detecting AI-generated speech with classical machine learning — no neural networks, 84 handcrafted acoustic features, and a result that only shows up when you stop looking at overall accuracy.

**The finding:** on re-recorded audio the SVM gets **89.4% of real clips right and 33.3% of fake ones**. Blend those and you get a respectable-looking 61% that hides the failure completely. Logistic Regression gives up 6 points on real audio to reach **81.5% on fake**, and is the only model that stays balanced under distribution shift.

This repository is the code behind the paper *Deepfake Audio Detection with Statistical Classifiers* (Bhavsar, Anand & Rathore, IIT Indore) — [`Report_PR_Final_paper.pdf`](Report_PR_Final_paper.pdf).

---

## Table of Contents

- [Results](#results)
- [Installation](#installation)
- [Getting the Dataset](#getting-the-dataset)
- [Running the Pipeline](#running-the-pipeline)
- [How It Works](#how-it-works)
- [Repository Layout](#repository-layout)
- [Reproducing the Paper](#reproducing-the-paper)
- [Known Limitations](#known-limitations)
- [Citation](#citation)

---

## Results

All numbers below are from a full run of this repository on the FoR dataset. They are reproducible with the commands in [Running the Pipeline](#running-the-pipeline).

### The key result — re-recorded audio (out of distribution)

`for-rerec/for-rerecorded/validation`, 1,101 real + 1,143 fake clips. Real and fake are scored **separately, by construction**.

| Model | Real accuracy | Fake accuracy | Balanced | Gap |
|---|---|---|---|---|
| **LR** | 83.74% | **81.54%** | **82.64%** | **2.20 pts** |
| LORIS9 (0.1 SVM / 0.9 LR) | 85.10% | 79.97% | 82.53% | 5.14 pts |
| LORIS7 (0.3 SVM / 0.7 LR) | 88.47% | 73.75% | 81.11% | 14.71 pts |
| SVM | **89.37%** | 33.25% | 61.31% | 56.13 pts |
| Random Forest | 78.11% | 37.62% | 57.87% | 40.49 pts |

The SVM is the best model on real audio and the second-worst on fake audio. It has learned to answer "human" — which is free accuracy on real clips and near-total failure on the class the system exists to catch.

### Held-out test split

`for-2sec/for-2seconds/testing`, 1,088 clips, balanced.

| Model | Accuracy | Weighted F1 | Real | Fake |
|---|---|---|---|---|
| Random Forest | 0.4982 | 0.4918 | 0.6103 | 0.3860 |
| SVM | 0.6360 | 0.5992 | 0.9393 | 0.3327 |
| LR | 0.6443 | 0.6421 | 0.7224 | 0.5662 |
| LORIS7 | 0.6553 | 0.6448 | 0.8272 | 0.4835 |
| LORIS9 | 0.6452 | 0.6410 | 0.7537 | 0.5368 |

Random Forest sits at chance level and was dropped from the paper's analysis. This run reproduces that decision.

### 5-fold cross-validation on the training split

| Model | Mean train accuracy | Mean validation accuracy |
|---|---|---|
| SVM | 1.0000 | 0.9898 |
| LR | 0.9131 | 0.9099 |

Note the gap: CV on the training split says 99% and 91%, but the held-out `for-2sec/testing` split says 64% for both. That is far larger than ordinary generalization loss and indicates the FoR testing split is already substantially out of distribution relative to training — before the re-recorded set is even introduced. This is discussed in [Known Limitations](#known-limitations).

### Ensemble weight sweep

Soft voting averages the two models' predicted probabilities, so there is exactly one knob:

```
P_final = w_LR × P_LR  +  (1 − w_LR) × P_SVM
```

| w_LR | w_SVM | Real acc | Fake acc | Macro F1 | Accuracy | Gap |
|---|---|---|---|---|---|---|
| 0.50 | 0.50 | 0.9081 | 0.4007 | 0.6306 | 0.6544 | 0.5074 |
| 0.55 | 0.45 | 0.8934 | 0.4283 | 0.6415 | 0.6608 | 0.4651 |
| 0.60 | 0.40 | 0.8640 | 0.4449 | 0.6385 | 0.6544 | 0.4191 |
| 0.65 | 0.35 | 0.8438 | 0.4798 | **0.6502** | **0.6618** | 0.3640 |
| 0.70 | 0.30 | 0.8272 | 0.4835 | 0.6448 | 0.6553 | 0.3437 |
| 0.75 | 0.25 | 0.8162 | 0.4982 | 0.6483 | 0.6572 | 0.3180 |
| 0.80 | 0.20 | 0.7923 | 0.5092 | 0.6436 | 0.6507 | 0.2831 |
| 0.85 | 0.15 | 0.7757 | 0.5257 | 0.6452 | 0.6507 | 0.2500 |
| 0.90 | 0.10 | 0.7537 | 0.5368 | 0.6410 | 0.6452 | **0.2169** |

As LR's weight rises, real accuracy falls and fake accuracy climbs — the gap narrows from 0.51 to 0.22. This is the SVM's bias being diluted.

### Takeaways

1. **Logistic Regression is the model to ship.** It is the only classifier that holds above 80% on *both* classes under distribution shift.
2. **The SVM is biased toward "real".** A detector that misses two-thirds of deepfakes has little practical value regardless of its headline score.
3. **The ensemble does not beat LR.** Both LORIS variants land between SVM and LR, inheriting some of the SVM's bias without gaining anything net.
4. **Per-class evaluation is not optional.** Every conclusion here comes from separating real and fake accuracy. A single blended number hides all of it.

---

## Installation

Requires **Python 3.10+** (developed and tested on 3.13).

```bash
git clone https://github.com/rahulkumarrathore/Deepfake-Audio-Detection-with-Statistical-Classifiers.git
cd Deepfake-Audio-Detection-with-Statistical-Classifiers

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```bash
python -c "import librosa, sklearn, pandas; print('ok')"
```

> **Note on scikit-learn.** `requirements.txt` pins `scikit-learn<1.10` deliberately. The paper's LR uses `penalty='l1'` (removed in 1.10) and the ensembles use `SVC(probability=True)` (removed in 1.11). Lifting the pin requires migrating to `l1_ratio=` and `CalibratedClassifierCV`, which changes the SVM's probability calibration and therefore shifts the ensemble numbers.

### PyCharm

Set the project interpreter to `.venv/bin/python`
(*Settings → Project → Python Interpreter → Add Local Interpreter → Select existing*).

There are **no command-line arguments** anywhere in this project — every script has a `SETTINGS` block at the top. Edit it, then hit Run. The Parameters field stays empty.

---

## Getting the Dataset

This project uses the **Fake-or-Real (FoR)** dataset from Kaggle (~19 GB total).

### Option A — the helper script

```bash
python scripts/download_dataset.py --check      # verify credentials + disk space
python scripts/download_dataset.py --subsets    # only for-2sec + for-rerec (~2.7 GB)
python scripts/download_dataset.py --full       # everything (~19 GB)
```

Kaggle needs an API token: [kaggle.com/settings](https://www.kaggle.com/settings) → API → *Create New API Token*, then

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Option B — download manually

Grab it from [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) and unzip it anywhere.

### Option C — no dataset at all

```bash
python scripts/make_demo_data.py
```

Generates 400 small **synthetic** clips under `data_demo/` so the whole pipeline runs end to end in about 3 seconds. Accuracy on it is meaningless — the real/fake difference is a caricature — but every stage executes and you can watch data move through. Point the pipeline at it with `FOR_DATA_ROOT=data_demo`.

### Telling the project where the data is

Three ways, in order of precedence:

```bash
# 1. environment variable (highest precedence)
export FOR_DATA_ROOT=/path/to/dataset

# 2. symlink the repo's data/ directory at it
ln -s /path/to/dataset data

# 3. edit DEFAULT_DATA_ROOT in src/config.py
```

### Expected layout

Only three directories are read. The other variants (`for-norm`, `for-original`) and splits are ignored.

```
<data root>/
├── for-2sec/for-2seconds/
│   ├── training/{real,fake}/*.wav      13,956 clips  ← training
│   └── testing/{real,fake}/*.wav        1,088 clips  ← held-out test
└── for-rerec/for-rerecorded/
    └── validation/{real,fake}/*.wav     2,244 clips  ← OOD benchmark
```

**Label convention:** the folder name *is* the label — `real → 0`, `fake → 1`. It is set in exactly one place, `get_class_from_path()` in `src/data_preprocessing.py`.

---

## Running the Pipeline

Three stages, run in order. Each is a plain `python -m` invocation with no flags.

### Stage 1 — extract features

```bash
python -m src.data_preprocessing
```

```
##### length of train#### 13956 (real 6978, fake 6978)
##### length of test #### 1088 (real 544, fake 544)
######### extracting features ###########
length of dict_audio_train: 13956
length of dict_audio_test: 1088
######### data structured ###########
train: (13956, 84) | test: (1088, 84)
```

There is **no feature cache**. Extraction runs on every invocation — about 66 seconds for 15,044 clips on an Apple M4. Running this stage directly is optional; Stage 2 imports it and triggers the same work.

### Stage 2 — train and evaluate

```bash
python -m src.train
```

Settings at the top of `src/train.py`:

```python
MODELS = ["rf", "svm", "lr", "ensemble"]   # any of: rf, svm, lr, ensemble
USE_CHROMA = True     # False -> drop the 12 chroma pairs (84-D -> 60-D)
DO_CV = True          # 5-fold cross-validation
DO_SWEEP = True       # sweep the LR ensemble weight 0.50 -> 0.90
DO_TUNE = False       # re-run GridSearchCV (slow)
SAVE = True           # write artifacts/models/*.joblib
```

Writes `artifacts/models/{rf,svm,lr,loris7,loris9}.joblib`. With `USE_CHROMA = False` the filenames gain a `_nochroma` suffix, so the 84-D and 60-D models coexist. Full run: **~110 seconds**.

### Stage 3 — inference and the OOD benchmark

```bash
python -m src.inference
```

Settings at the top of `src/inference.py`:

```python
MODELS = ["rf", "svm", "lr", "loris7", "loris9"]
MODE = "rerec"        # "rerec" | "file" | "dir"
PREDICT_FILE = ""     # used when MODE == "file"
PREDICT_DIR = ""      # used when MODE == "dir"
USE_CHROMA = True
```

- `"rerec"` — the out-of-distribution benchmark, scored per class (~12 s)
- `"file"` — predict one audio file
- `"dir"` — predict every audio file in a directory

Each saved model is a complete `Pipeline` containing its own fitted `StandardScaler`, so it predicts on new audio with no other setup.

### Total runtime

| Stage | Apple M4 (10-core, 16 GB) |
|---|---|
| Feature extraction, 15,044 clips | ~66 s |
| Train RF + SVM + LR + 2 ensembles + CV + sweep | ~110 s |
| Re-recorded benchmark, 2,244 clips | ~12 s |

No GPU required. Peak memory is well under 1 GB — the feature matrix is 13,956 × 84 floats.

---

## How It Works

### Feature extraction

Every clip becomes a fixed-length **84-dimensional vector**, regardless of its duration.

Audio is loaded at 16 kHz, then seven feature families are computed with `librosa`:

| Family | Coefficients |
|---|---|
| MFCCs | 13 |
| Δ MFCCs (first-order derivatives) | 13 |
| Chroma (STFT) | 12 |
| Spectral centroid | 1 |
| Spectral bandwidth | 1 |
| Zero-crossing rate | 1 |
| RMS energy | 1 |
| **Total** | **42** |

Each call returns a `(n_coefficients, n_frames)` matrix whose width depends on clip length — so a 2-second clip and a 5-second clip produce differently shaped matrices, and an SVM cannot consume either. Taking the **mean and standard deviation along the time axis** collapses that dimension:

```python
for family in families:
    features.extend(np.mean(family, axis=1))
    features.extend(np.std(family, axis=1))
```

**42 series × 2 statistics = 84 features.** The cost is that all temporal structure is discarded: "high pitch then low" and "low then high" produce identical vectors.

### Pipeline

```
 FoR dataset
     │
     ├── for-2sec/training    ─┐
     ├── for-2sec/testing     ─┤  load_audio() → extract_features() → 84-D rows → DataFrame
     └── for-rerec/validation ─┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
              X_train / y_train              X_test / y_test
                     │
        Pipeline([StandardScaler, classifier])
                     │
     ┌───────────────┼───────────────┬───────────────┐
     RF             SVM              LR         VotingClassifier
                     │                │          (soft, weighted)
                     └────────┬───────┘                │
                              │                   loris7 / loris9
                     artifacts/models/*.joblib
                              │
                  for-rerec/validation  →  per-class accuracy
```

### Models

| Model | Configuration | Source |
|---|---|---|
| Random Forest | `n_estimators=100` | Baseline; dropped (chance level) |
| **SVM (RBF)** | `C=10, gamma=0.01` | `GridSearchCV`, `cv=5`, `scoring='f1'` |
| **Logistic Regression** | `C=0.1, penalty='l1', solver='liblinear', max_iter=500` | `GridSearchCV`, `cv=5`, `scoring='f1'` |
| LORIS7 | Soft-vote ensemble — 0.3 SVM / 0.7 LR | Weight sweep |
| LORIS9 | Soft-vote ensemble — 0.1 SVM / 0.9 LR | Weight sweep |

The digit in LORIS7 / LORIS9 **is** the LR weight × 10. The paper never expands the acronym.

`_nochroma` variants of any model are produced by setting `USE_CHROMA = False`.

### Design decisions

- **The scaler lives inside the Pipeline.** A saved `.joblib` is self-contained and predicts without any other setup. Cross-validation also re-fits the scaler inside every fold, so no validation statistics leak into scaling.
- **One feature-extraction function.** `extract_features()` in `src/features.py` is the only place features are computed; training and inference both call it, so they cannot drift apart.
- **`src/features.py` does no module-level work.** `src/data_preprocessing.py` builds the whole dataset on import — which is why inference imports the former, not the latter, and does not trigger a 15,000-file extraction to classify one clip.
- **The weight sweep fits each model once.** Rebuilding a `VotingClassifier` per weight would refit the SVM nine times. The sweep stores each model's probability matrix and applies the weights to that instead — mathematically identical to what sklearn's soft voting computes.
- **`chroma_stft(tuning=0.0)`.** By default librosa estimates a musical tuning offset per file, which costs about four times as much as the chroma computation itself, emits warnings on clipped clips, and gives every file its *own* correction. A fixed tuning treats every clip alike; values shift by under 0.01.

---

## Repository Layout

```
.
├── src/
│   ├── config.py               paths, hyperparameters, label constants
│   ├── dataset_directory.py    file-path lists (glob + natsort); no audio is read
│   ├── features.py             load_audio() + extract_features(); no module-level work
│   ├── data_preprocessing.py   STAGE 1: audio -> 84-D table (builds on import)
│   ├── train.py                STAGE 2: features -> trained Pipelines + weight sweep
│   └── inference.py            STAGE 3: saved model + new audio -> prediction
├── scripts/
│   ├── download_dataset.py     fetch FoR from Kaggle, with preflight checks
│   └── make_demo_data.py       400 synthetic clips, to run without the real dataset
├── artifacts/models/           saved pipelines (.joblib, gitignored)
├── Report_PR_Final_paper.pdf   the paper this code implements
├── Fake_Audio_Detection_Project.ipynb    original Colab notebook (143 cells)
└── requirements.txt
```

The notebook is the original research artifact. `src/` is that same pipeline split into stages so each can be read and run on its own — and with several defects fixed along the way (see [Known Limitations](#known-limitations)).

---

## Reproducing the Paper

A full run of this repository against the paper's reported figures:

| Quantity | Paper | This repo |
|---|---|---|
| SVM accuracy (test) | 63.24% | 63.60% |
| LR accuracy (test) | 63.69% | 64.43% |
| Random Forest accuracy | 45.13% | 49.82% |
| SVM real → fake drop (re-recorded) | ~67% | 65% (89.37 → 33.25) |
| LORIS7 vs LR on real | +5% | +4.73 pts |
| LORIS7 vs LR on fake | −9% | −7.79 pts |
| LORIS9 vs LR on real | +1% | +1.36 pts |
| LORIS9 vs LR on fake | −2% | −1.57 pts |

The ensemble construction code was **missing** from the original notebook — `loris1` and `loris2` survived only as `pickle.load()` calls. `make_ensemble()` in `src/train.py` is a reconstruction from the paper's description, and the agreement in the table above is the evidence that the reconstruction is correct.

**Where 0.7 and 0.9 come from.** The paper selects these two weights from a sweep (its Figure 4) but does not state the criterion. Running `DO_SWEEP = True` reproduces the sweep: `w_LR = 0.90` is justified — it is the most balanced configuration — but `w_LR = 0.70` is not optimal on any metric in the table; `0.65` beats it on both macro F1 and accuracy.

---

## Known Limitations

- **Cross-validation and test accuracy disagree sharply.** 5-fold CV on the training split gives SVM 98.98% and LR 90.99%, while the same models score 63.6% and 64.4% on the held-out `for-2sec/testing` split. A ~35-point gap is larger than ordinary generalization loss and indicates the testing split differs in distribution from training. The paper does not discuss this.
- **No temporal modelling.** Collapsing every feature to a mean and a standard deviation over time discards all sequential structure. This is what makes fixed-length classical classifiers possible at all, and it is also the ceiling on this approach.
- **The CNN baseline was never run.** The notebook contains a complete unexecuted PyTorch raw-waveform CNN (cell 142, titled `#CNN - cancel not doing`). The comparison against a deep learning baseline remains future work.
- **Ensemble weight selection is unjustified.** See [Reproducing the Paper](#reproducing-the-paper).
- **Two deprecations are pending.** `LogisticRegression(penalty=)` is removed in scikit-learn 1.10 and `SVC(probability=True)` in 1.11. `requirements.txt` pins below 1.10; migrating changes the SVM's probability calibration and will shift the ensemble numbers.
- **Feature extraction is not cached.** By design — the pipeline is meant to be read and traced end to end — but it does mean 66 seconds of recomputation on every training run.

### Fixed relative to the original notebook

- The `StandardScaler` was fitted, used, and never saved, so a pickled model could not predict on its own. It now lives inside the `Pipeline`.
- The scaler was fitted on the full training set *before* cross-validation, leaking each fold's validation statistics into scaling. The `Pipeline` re-fits it per fold.
- Three near-identical copies of the feature-extraction function existed; a `use_chroma` flag replaced all three.
- Column names were generated in interleaved order while values were emitted in block order, so labels did not describe the values beneath them. Accuracy was unaffected (a consistent permutation), but any per-feature interpretation would have been wrong.
- The re-recorded evaluation ran 12 loops over the same 2,244 files (~27,000 extractions where 2,244 suffice). Features are now extracted once and reused across models.
- All Google Drive paths are gone; the data root is configurable.


**Dataset:** [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) (Kaggle)
