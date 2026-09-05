"""
STAGE 2 -- TRAINING & EVALUATION
================================

Imports Stage 1 (which extracts features on the spot), trains the classifiers,
evaluates them, and saves each one as a self-contained sklearn Pipeline.

    features from src.data_preprocessing
        -> split into X (84 features) and y (label)
        -> Pipeline([StandardScaler, classifier])
        -> optional GridSearchCV hyperparameter search
        -> 5-fold cross-validation on the training set
        -> evaluation on the held-out test split, reported PER CLASS
        -> joblib.dump(pipeline)  <- scaler travels with the model

Why a Pipeline instead of a bare classifier?
    Two reasons, both fixes for real problems in the original notebook.

    1. The notebook fitted a StandardScaler, used it, and never saved it.
       A pickled model was therefore useless on its own -- you had to rebuild
       the scaler from the training data before you could predict anything.
       Inside a Pipeline the scaler is part of the saved artifact.

    2. The notebook fitted the scaler on the FULL training set and then ran
       cross-validation on the already-scaled data. That leaks each fold's
       validation statistics into the scaling step. A Pipeline re-fits the
       scaler inside every fold, so the CV score is honest.

No command-line arguments -- edit the SETTINGS block below and hit Run.

    python -m src.train
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.data_preprocessing import (X_train as X_train_full, X_test as X_test_full,
                                    y_train as y_train_full, y_test as y_test_full,
                                    drop_chroma)


# ==========================================================================
# SETTINGS -- change these, then just Run. No command-line flags.
# ==========================================================================
MODELS = ["rf", "svm", "lr", "ensemble"]   # any of: rf, svm, lr, ensemble

USE_CHROMA = True     # False -> drop the 12 chroma pairs (84-D -> 60-D).
                      # Extraction always produces 84; this drops columns,
                      # and saves the models under a "_nochroma" suffix.

DO_CV = True          # 5-fold cross-validation on the training split
DO_SWEEP = True       # Figure 4: sweep the LR ensemble weight 0.50 -> 0.90
DO_TUNE = False       # re-run GridSearchCV for svm/lr (slow)
SAVE = True           # write artifacts/models/*.joblib


# ==========================================================================
# 1. Model definitions -- every model is scaler + classifier
# ==========================================================================
def make_pipeline(kind: str) -> Pipeline:
    """Build an unfitted Pipeline for the requested model."""
    if kind == "svm":
        # probability=True is needed for the soft-voting ensemble below.
        # It costs extra fit time (internal Platt-scaling cross-validation)
        # but does not change the decision boundary.
        clf = SVC(**config.SVM_PARAMS, probability=True, random_state=config.RANDOM_STATE)
    elif kind == "lr":
        clf = LogisticRegression(**config.LR_PARAMS, random_state=config.RANDOM_STATE)
    elif kind == "rf":
        clf = RandomForestClassifier(**config.RF_PARAMS)
    else:
        raise ValueError(f"Unknown model kind: {kind!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_ensemble(weights: dict) -> Pipeline:
    """
    Weighted soft-vote ensemble of SVM and LR -- the notebook's "LORIS".

    RECONSTRUCTED. The cells that built loris1/loris2 were deleted from the
    notebook; only the pickle.load() calls and the reported numbers survive.
    This averages predicted probabilities with the documented weights, which
    is the standard reading of "0.3 SVM, 0.7 LR". Results from this code may
    not reproduce the notebook's LORIS numbers exactly.
    """
    svm = SVC(**config.SVM_PARAMS, probability=True, random_state=config.RANDOM_STATE)
    lr = LogisticRegression(**config.LR_PARAMS, random_state=config.RANDOM_STATE)
    voter = VotingClassifier(
        estimators=[("svm", svm), ("lr", lr)],
        voting="soft",
        weights=[weights["svm"], weights["lr"]],
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", voter)])


# ==========================================================================
# 2. Evaluation helpers
# ==========================================================================
def evaluate(pipe, X, y, name: str) -> dict:
    """Score a fitted pipeline and print a per-class breakdown."""
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    f1w = f1_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred, labels=[config.LABEL_REAL, config.LABEL_FAKE])

    # Per-class accuracy = recall. This is the number that matters: a model
    # can post a good overall score while being nearly blind to one class.
    real_acc = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    fake_acc = cm[1, 1] / cm[1].sum() if cm[1].sum() else float("nan")

    print(f"\n--- {name} ---")
    print(f"  accuracy      : {acc:.4f}")
    print(f"  weighted F1   : {f1w:.4f}")
    print(f"  real accuracy : {real_acc:.4f}   ({cm[0, 0]}/{cm[0].sum()})")
    print(f"  fake accuracy : {fake_acc:.4f}   ({cm[1, 1]}/{cm[1].sum()})")
    print(f"  confusion matrix [rows=true real,fake | cols=pred real,fake]:\n{cm}")
    print(classification_report(y, y_pred,
                                target_names=["real (0)", "fake (1)"],
                                zero_division=0))
    return {"name": name, "accuracy": acc, "weighted_f1": f1w,
            "real_accuracy": real_acc, "fake_accuracy": fake_acc}


def cross_validate_model(pipe, X, y, name: str) -> dict:
    """5-fold CV on the training set. The scaler is re-fitted inside each fold."""
    res = cross_validate(pipe, X, y, cv=config.CV_FOLDS,
                         return_train_score=True, n_jobs=-1)
    train_mean = float(np.mean(res["train_score"]))
    val_mean = float(np.mean(res["test_score"]))
    print(f"\n--- {name}: {config.CV_FOLDS}-fold cross-validation ---")
    print(f"  train accuracy per fold : {np.round(res['train_score'], 4)}")
    print(f"  val   accuracy per fold : {np.round(res['test_score'], 4)}")
    print(f"  mean train accuracy     : {train_mean:.4f}")
    print(f"  mean val   accuracy     : {val_mean:.4f}")
    return {"cv_train_mean": train_mean, "cv_val_mean": val_mean}


def sweep_ensemble(X_train, y_train, X_eval, y_eval, lr_weights=None,
                   eval_name="test"):
    """
    Figure 4 of the paper: WHERE 0.7 AND 0.9 COME FROM.

    Soft voting is just a weighted average of the two models' predicted
    probabilities:

        P_final = w_LR * P_LR  +  (1 - w_LR) * P_SVM

    so w_LR is the only knob. This sweeps it and reports, at each value, the
    four quantities the paper plots -- per-class score, macro F1, accuracy --
    and you read the trade-off off the table.

    IMPORTANT: SVM and LR are fitted ONCE, then the weights are applied to
    their stored probability matrices. Rebuilding a VotingClassifier per
    weight would refit both models nine times; on 14k samples the SVM fit
    alone is tens of minutes, so that would turn a 30-minute job into a
    5-hour one. The result is identical -- sklearn's soft voting computes
    exactly this weighted average.
    """
    if lr_weights is None:
        lr_weights = config.ENSEMBLE_SWEEP

    print("\n" + "=" * 70)
    print(f"ENSEMBLE WEIGHT SWEEP  (evaluated on the {eval_name} split)")
    print("=" * 70)
    print("Fitting SVM and LR once ...")
    svm = make_pipeline("svm").fit(X_train, y_train)
    lr = make_pipeline("lr").fit(X_train, y_train)

    # Both pipelines saw the same y, so their class order matches; the columns
    # of the two probability matrices line up and can be averaged directly.
    assert list(svm.classes_) == list(lr.classes_) == [config.LABEL_REAL,
                                                       config.LABEL_FAKE]
    p_svm = svm.predict_proba(X_eval)
    p_lr = lr.predict_proba(X_eval)

    rows = []
    for w_lr in lr_weights:
        proba = w_lr * p_lr + (1 - w_lr) * p_svm
        y_pred = svm.classes_[np.argmax(proba, axis=1)]

        cm = confusion_matrix(y_eval, y_pred,
                              labels=[config.LABEL_REAL, config.LABEL_FAKE])
        f1_per_class = f1_score(y_eval, y_pred, average=None,
                                labels=[config.LABEL_REAL, config.LABEL_FAKE],
                                zero_division=0)
        rows.append({
            "w_lr": w_lr,
            "w_svm": round(1 - w_lr, 2),
            # per-class accuracy = recall = the paper's "human/AI class accuracy"
            "human_acc": cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan"),
            "ai_acc": cm[1, 1] / cm[1].sum() if cm[1].sum() else float("nan"),
            # per-class F1 = what Figure 4's legend actually names
            "human_f1": f1_per_class[0],
            "ai_f1": f1_per_class[1],
            "macro_f1": f1_score(y_eval, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_eval, y_pred),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'w_LR':>6}{'w_SVM':>7}{'human_acc':>11}{'ai_acc':>9}"
          f"{'human_f1':>10}{'ai_f1':>8}{'macro_f1':>10}{'accuracy':>10}{'gap':>8}")
    for r in rows:
        gap = abs(r["human_acc"] - r["ai_acc"])
        print(f"{r['w_lr']:>6.2f}{r['w_svm']:>7.2f}{r['human_acc']:>11.4f}"
              f"{r['ai_acc']:>9.4f}{r['human_f1']:>10.4f}{r['ai_f1']:>8.4f}"
              f"{r['macro_f1']:>10.4f}{r['accuracy']:>10.4f}{gap:>8.4f}")

    best_macro = df.loc[df["macro_f1"].idxmax()]
    best_bal = df.loc[(df["human_acc"] - df["ai_acc"]).abs().idxmin()]
    print(f"\n  best macro F1      : w_LR={best_macro['w_lr']:.2f}  "
          f"(macro_f1={best_macro['macro_f1']:.4f})")
    print(f"  most balanced      : w_LR={best_bal['w_lr']:.2f}  "
          f"(human {best_bal['human_acc']:.4f} vs ai {best_bal['ai_acc']:.4f})")
    print("\nThe paper picked 0.70 (LORIS7) and 0.90 (LORIS9) off this sweep but\n"
          "never stated the criterion. Compare those rows against the two above.")
    return df


def tune(kind: str, X, y):
    """Re-run the hyperparameter search. Slow -- this is what produced the
    values sitting in config.SVM_PARAMS / config.LR_PARAMS."""
    grid = {"svm": config.SVM_GRID, "lr": config.LR_GRID}[kind]
    print(f"\nGridSearchCV for {kind} over {grid} ...")
    search = GridSearchCV(make_pipeline(kind), param_grid=grid,
                          cv=config.CV_FOLDS, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(X, y)
    print(f"  best params : {search.best_params_}")
    print(f"  best F1     : {search.best_score_:.4f}")
    return search.best_estimator_


# ==========================================================================
# 3. Main training routine
# ==========================================================================
def model_path(name: str, use_chroma: bool = True):
    suffix = "" if use_chroma else "_nochroma"
    return config.MODELS_DIR / f"{name}{suffix}.joblib"


def run(models, use_chroma=True, do_tune=False, do_cv=True, save=True,
        do_sweep=False):
    use_str = "84-D (full)" if use_chroma else "60-D (chroma removed)"
    print("=" * 70)
    print(f"TRAINING  |  feature set: {use_str}")
    print("=" * 70)

    # ---- Stage 1 output: already built when data_preprocessing was imported ----
    if not use_chroma:
        X_train, X_test = drop_chroma(X_train_full), drop_chroma(X_test_full)
    else:
        X_train, X_test = X_train_full, X_test_full
    y_train, y_test = y_train_full, y_test_full

    print(f"train: {X_train.shape[0]} samples x {X_train.shape[1]} features")
    print(f"test : {X_test.shape[0]} samples x {X_test.shape[1]} features")
    print(f"train label balance: {dict(y_train.value_counts())}")

    if do_sweep:
        sweep_ensemble(X_train, y_train, X_test, y_test, eval_name="test")

    results = []
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in models:
        t0 = time.time()

        if name == "ensemble":
            for ens_name, weights in config.ENSEMBLE_WEIGHTS.items():
                pipe = make_ensemble(weights)
                pipe.fit(X_train, y_train)
                label = f"{ens_name} (svm={weights['svm']}, lr={weights['lr']})"
                row = evaluate(pipe, X_test, y_test, label)
                results.append(row)
                if save:
                    joblib.dump(pipe, model_path(ens_name, use_chroma))
                    print(f"  saved -> {model_path(ens_name, use_chroma)}")
            continue

        pipe = tune(name, X_train, y_train) if do_tune and name in ("svm", "lr") \
            else make_pipeline(name)

        if do_cv:
            cross_validate_model(pipe, X_train, y_train, name.upper())

        pipe.fit(X_train, y_train)
        row = evaluate(pipe, X_test, y_test, name.upper())
        row["fit_seconds"] = round(time.time() - t0, 1)
        results.append(row)

        if save:
            joblib.dump(pipe, model_path(name, use_chroma))
            print(f"  saved -> {model_path(name, use_chroma)}")

    # ---- summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY (held-out test split)")
    print("=" * 70)
    print(f"{'model':<32}{'acc':>8}{'wF1':>8}{'real':>8}{'fake':>8}")
    for r in results:
        print(f"{r['name']:<32}{r['accuracy']:>8.4f}{r['weighted_f1']:>8.4f}"
              f"{r['real_accuracy']:>8.4f}{r['fake_accuracy']:>8.4f}")
    return results


if __name__ == "__main__":
    run(models=MODELS, use_chroma=USE_CHROMA, do_tune=DO_TUNE,
        do_cv=DO_CV, save=SAVE, do_sweep=DO_SWEEP)
