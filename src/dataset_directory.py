"""
Path lists for the Fake-or-Real dataset.

Nothing is loaded here -- this module only collects file paths, so importing it
is instant. The audio itself is read in data_preprocessing.py.

Layout expected under BASE_DIR_WAV:
    for-2sec/for-2seconds/training/{real,fake}/*.wav
    for-2sec/for-2seconds/testing/{real,fake}/*.wav
    for-rerec/for-rerecorded/validation/{real,fake}/*.wav
"""

import os
from glob import glob

import natsort

from src import config

# Set in config.py (DEFAULT_DATA_ROOT), overridable with the FOR_DATA_ROOT
# environment variable.
BASE_DIR_WAV = str(config.DATA_ROOT)

TRAIN_DIR = os.path.join(BASE_DIR_WAV, "for-2sec", "for-2seconds", "training")
TEST_DIR = os.path.join(BASE_DIR_WAV, "for-2sec", "for-2seconds", "testing")
REREC_DIR = os.path.join(BASE_DIR_WAV, "for-rerec", "for-rerecorded", "validation")


def glob_audio(folder):
    """All audio files in `folder`, natural-sorted. The FOR dataset is .wav,
    but the rerecorded subset also ships .mp3 and .webm."""
    paths = []
    for ext in config.AUDIO_EXTENSIONS:
        paths += glob(os.path.join(folder, f"*{ext}"))
    return natsort.natsorted(paths)


# --------------------------------------------------------------------------
# training split
# --------------------------------------------------------------------------
list_wav_paths_train_real = glob_audio(os.path.join(TRAIN_DIR, "real"))
list_wav_paths_train_fake = glob_audio(os.path.join(TRAIN_DIR, "fake"))
list_wav_paths_train = list_wav_paths_train_real + list_wav_paths_train_fake

# --------------------------------------------------------------------------
# testing split
# --------------------------------------------------------------------------
list_wav_paths_test_real = glob_audio(os.path.join(TEST_DIR, "real"))
list_wav_paths_test_fake = glob_audio(os.path.join(TEST_DIR, "fake"))
list_wav_paths_test = list_wav_paths_test_real + list_wav_paths_test_fake

# --------------------------------------------------------------------------
# rerecorded validation split (the out-of-distribution benchmark)
# --------------------------------------------------------------------------
list_wav_paths_rerec_real = glob_audio(os.path.join(REREC_DIR, "real"))
list_wav_paths_rerec_fake = glob_audio(os.path.join(REREC_DIR, "fake"))
list_wav_paths_rerec = list_wav_paths_rerec_real + list_wav_paths_rerec_fake

print("##### base dir       ####", BASE_DIR_WAV)
print("##### length of train####", len(list_wav_paths_train),
      f"(real {len(list_wav_paths_train_real)}, fake {len(list_wav_paths_train_fake)})")
print("##### length of test ####", len(list_wav_paths_test),
      f"(real {len(list_wav_paths_test_real)}, fake {len(list_wav_paths_test_fake)})")
print("##### length of rerec####", len(list_wav_paths_rerec),
      f"(real {len(list_wav_paths_rerec_real)}, fake {len(list_wav_paths_rerec_fake)})")
