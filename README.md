# CS4120FinalProject

IMDb Multi-Label Genre Classification
Course: CS4120 Natural Language Processing
Group Members: Amulya Dussa, Eftyghia Kourtelidis, Maxim Onashchenko, John Young

# Project Overview
This project tackles multi-label genre classification of movies using NLP techniques applied to IMDb plot summaries. Given a textual description of a film, our system predicts one or more genre labels from a set of 27 categories (e.g. Drama, Comedy, Horror, Thriller). We build and compare a series of models of increasing sophistication — a TF-IDF + Logistic Regression baseline, a custom feedforward neural network, and a fine-tuned BERT transformer (in progress) — evaluating each using macro-averaged F1, per-genre precision/recall, and error analysis.

# Dataset
We use the IMDb Genre Classification Dataset, available on Kaggle. It contains approximately 54,000 English-language movie plot summaries, each labeled with one or more genres across 27 categories.
Download the dataset and place the files in the dataset/ folder:
dataset/
├── train_data.txt
├── test_data.txt
└── test_data_solution.txt

# Dependencies
Install all required packages with:
bashpip install numpy pandas matplotlib seaborn scikit-learn joblib torch nltk

# How to Run
The three scripts must be run in order. Each script depends on the outputs of the previous one.
Step 1 — Preprocessing
bashpython preprocessing_nlp.py
Author: Amulya Dussa
Loads and cleans the raw plot summaries, performs tokenization, stopword removal, and lemmatization via NLTK, encodes genre labels using MultiLabelBinarizer, and creates TF-IDF feature matrices. Saves all processed data and fitted transformers to the models/ folder as .joblib files.
Outputs to models/:

mlb.joblib — fitted MultiLabelBinarizer
tfidf_vectorizer.joblib — fitted TF-IDF vectorizer
X_train.joblib, X_val.joblib, y_train.joblib, y_val.joblib — train/val splits
X_test_tfidf.joblib — test features
train_processed.pkl, test_processed.pkl, test_solution_processed.pkl — processed dataframes


Step 2 — Modeling
bashpython models.py
Author: Maxim Onashchenko
Trains a Logistic Regression baseline (One-vs-Rest with TF-IDF features) and a custom feedforward neural network (PyTorch) with sigmoid output and binary cross-entropy loss. Includes threshold tuning to optimize predicted genre label assignments.
Outputs to models/:

lr_model.joblib — trained Logistic Regression model
best_nn.pt — best neural network weights
model_comparison.png — loss curves and micro F1 comparison chart


Step 3 — Evaluation
bashpython evaluation.py
Author: John Young
Loads both trained models and runs full evaluation on the validation set including class weight analysis, per-genre Precision/Recall/F1 bar charts, per-genre confusion heatmaps, and FP/FN error analysis for both models.
Outputs to models/:

per_genre_metrics_Logistic_Regression.png
per_genre_metrics_Neural_Network.png
confusion_heatmap_Logistic_Regression.png
confusion_heatmap_Neural_Network.png
error_analysis_Logistic_Regression.png
error_analysis_Neural_Network.png

Also prints class weights and ranked error analysis tables to the terminal.

# Repository Structure
CS4120FinalProject/
├── dataset/
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
├── models/               ← generated at runtime, not tracked by git
├── preprocessing_nlp.py  ← Step 1: data cleaning and feature extraction
├── models.py             ← Step 2: model training
├── evaluation.py         ← Step 3: evaluation and error analysis
└── README.md

# Results Summary
ModelMicro F1Macro F1Logistic Regression0.49510.1689Neural Network0.60030.2433
The neural network substantially outperforms the LR baseline, particularly in recall for mid-frequency genres like horror, comedy, and short. Both models struggle with rare genres (e.g. war, news, history, musical) due to severe class imbalance — motivating the class weighting analysis in evaluation.py.

# Notes
The models/ folder is generated at runtime and should not be committed to the repo. Add it to .gitignore.
NLTK data packages (punkt, punkt_tab, stopwords, wordnet) must be available locally. They will download automatically on first run.
Neural network training runs on CPU by default on most machines and takes approximately 10–15 minutes.
