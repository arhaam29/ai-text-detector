# ai-text-detector
# SEA820 NLP Final Project

```
Arhaam Khan
ana40@myseneca.ca
Seneca Polytechnic
```

## Overview
This project aims to classify text as **human-written** or **AI-generated** using two different approaches:
1. **Classic Machine Learning** — TF-IDF + Logistic Regression
2. **Transformer-based Fine-Tuning** — DistilBERT from Hugging Face Transformers

We compare the models in terms of accuracy, precision, recall, and F1-score.

---

## Dataset
- **Source:** Kaggle — AI vs Human Text Dataset
- **Labels: Generated Column**
  - `0.0` → Human-written
  - `1.0` → AI-generated
- Dataset was preprocessed with tokenization, stopword removal, punctuation stripping, and lemmatization.
- For Transformer fine-tuning, a **stratified 5,000-row sample** was used for efficiency.

---

## Methodology

### 1. Classic Model (TF-IDF + Logistic Regression)
- Preprocessing: Tokenization, stopword removal, punctuation removal, lemmatization
- Feature extraction: TF-IDF vectorization
- Classifier: Logistic Regression (`scikit-learn`)
- Experiments with **full dataset** and **5K stratified subset**

### 2. Transformer Model (DistilBERT)
- Tokenization using `AutoTokenizer` from Hugging Face
- Fine-tuned for binary classification using `Trainer` API
- Used stratified 5K sample for faster training

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression (Full Dataset) | 0.95 | 0.97 | 0.89 | 0.93 |
| TF-IDF + Logistic Regression (5K Subset) | 0.87 | 1.00 | 0.65 | 0.79|
| DistilBERT (5K Subset) | 0.97 | X.XX | X.XX | 0.96 |

---

## Project Strcuture
```
├── NB1_classic_model.ipynb # Full dataset TF-IDF Logistic Regression
├── NB3_StratifiedSample_TF_IDF.ipynb # TF-IDF Logistic Regression on stratified 5K subset
├── NB2_Transformer_Model.ipynb # DistilBERT fine-tuning on stratified 5K subset
└──  README.md # Project description & usage
```

## How to run
```
1. Open the notebooks in colab.
2. Run each cell using Shift+Enter or manually run the cells
```
