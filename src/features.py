"""
Feature extraction -- one audio signal in, one fixed-length vector out.

Kept in its own module with NO module-level work, so inference.py can import it
without triggering the full dataset build in data_preprocessing.py.

    signal (any length)
        -> 7 feature families, each a (n_coefficients, n_frames) matrix
        -> mean & std over the TIME axis
        -> 84 numbers (60 when chroma is dropped)

Why mean/std over time?
    Each librosa feature is a matrix whose width depends on clip length, so
    clips of different durations produce different-sized matrices. SVM and LR
    need a fixed-length input. Collapsing the time axis into two summary
    statistics per coefficient gives every clip the same 84 numbers, at the
    cost of discarding all temporal structure.
"""

import librosa
import numpy as np

from src import config


def load_audio(path_wav):
    """Read an audio file and resample to 16 kHz. Returns (signal, sr)."""
    signal, sr = librosa.load(path_wav, sr=config.SAMPLE_RATE)
    return signal, sr


def extract_features(signal, sr, use_chroma=True):
    """
    84 floats for one clip (60 when use_chroma=False).

    This is the single source of truth for feature extraction -- training and
    inference MUST call this same function, or the vectors will not line up
    with the model.
    """
    # --- the 7 feature families, each shaped (n_coefficients, n_frames) ---
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=config.N_MFCC)  # (13, T)
    delta_mfccs = librosa.feature.delta(mfccs)                           # (13, T)
    spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)       # (1,  T)
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)        # (1,  T)
    # tuning=0.0 skips librosa's per-file tuning estimation. That estimation
    # is ~4x the cost of chroma itself, warns on clipped clips it cannot find
    # pitch bins in, and gives every file its OWN correction -- so identical
    # audio in two files could come out with slightly different chroma. A
    # fixed tuning treats every clip the same. The values shift by <0.01.
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, tuning=0.0)    # (12, T)
    zcr = librosa.feature.zero_crossing_rate(signal)                     # (1,  T)
    rms = librosa.feature.rms(y=signal)                                  # (1,  T)

    families = [mfccs, delta_mfccs, spec_cent, spec_bw]
    if use_chroma:
        families.append(chroma)
    families += [zcr, rms]

    # --- collapse the time axis: all means for a family, then all stds ---
    features = []
    for family in families:
        features.extend(np.mean(family, axis=1))
        features.extend(np.std(family, axis=1))
    return features


def extract_features_from_path(path_wav, use_chroma=True):
    """Convenience wrapper: load a file and extract its vector."""
    signal, sr = load_audio(path_wav)
    return extract_features(signal, sr, use_chroma)


def feature_names(use_chroma=True):
    """
    Column names, in the exact order extract_features() emits values.

    Names are in BLOCK order (all means for a family, then all stds), which is
    the order the values actually come out in. The original notebook generated
    them INTERLEAVED (mfcc0_mean, mfcc0_std, mfcc1_mean, ...), so its column
    labels did not describe the values underneath them.
    """
    families = [
        ([f"mfcc{i}" for i in range(config.N_MFCC)], True),
        ([f"delta_mfcc{i}" for i in range(config.N_MFCC)], True),
        (["spec_cent"], True),
        (["spec_bw"], True),
        ([f"chroma{i}" for i in range(config.N_CHROMA)], use_chroma),
        (["zcr"], True),
        (["rms"], True),
    ]
    names = []
    for base, include in families:
        if not include:
            continue
        names += [f"{b}_mean" for b in base]   # all means first ...
        names += [f"{b}_std" for b in base]    # ... then all stds
    return names
