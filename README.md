
# Deepfake Audio Detection using Statistical Machine Learning Models

This repository presents a lightweight and interpretable approach for **AI-generated (deepfake) audio detection** using classical statistical machine learning models. The work focuses on **Logistic Regression (LR)** and **Support Vector Machine (SVM)** classifiers and evaluates their performance against a CNN baseline under limited-data conditions.

This repository accompanies the project:
**"Deepfake Audio Detection with Statistical Classifiers"**

# Motivation

Recent advances in AI-driven voice synthesis and cloning technologies have made it increasingly difficult to distinguish real human speech from synthetic audio. While deep learning approaches achieve strong results, they often require large datasets and significant computational resources.

This project demonstrates that **statistical models**:
- Are computationally efficient
- Offer better interpretability
- Perform competitively on small-to-medium datasets

# Models Used

- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Weighted Ensemble (LR + SVM)
- CNN baseline (for comparison)


# Dataset

**Fake-or-Real (FoR) Dataset**
- ~14,000 audio samples
- Balanced real and AI-generated speech
- Subsets used:
  - `for-2second` (training & validation)
  - `for-rerecorded` (robustness testing)


# Feature Extraction

Each audio sample is represented using **84 handcrafted features**, including:

- 13 MFCCs
- Δ MFCCs (first-order derivatives)
- Chroma features
- Spectral centroid
- Spectral bandwidth
- Zero Crossing Rate (ZCR)
- RMS energy

Mean and standard deviation are computed per feature → **84-D feature vector**

# Experiments Conducted

- Hyperparameter tuning using GridSearchCV
- Feature reduction analysis
- Class-wise accuracy analysis
- Ensemble learning evaluation
- Comparison with CNN baseline


# Key Findings

- Logistic Regression provides **most balanced performance**
- SVM shows strong bias toward real audio
- Ensemble does **not outperform LR**
- Statistical models are competitive in low-data settings
