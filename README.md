CS4120FinalProject — IMDb Multi-Label Genre Classification
Course: CS4120 Natural Language Processing
Group Members: Amulya Dussa, Eftyghia Kourtelidis, Maxim Onashchenko, John Young
Project Overview
This project tackles multi-label genre classification of movies using NLP techniques applied to IMDb plot summaries. Given a textual description of a film, our system predicts one or more genre labels from a set of 27 categories (e.g. Drama, Comedy, Horror, Thriller). We build and compare a TF-IDF + Logistic Regression baseline, a custom feedforward neural network, and a fine-tuned BERT transformer (in progress), evaluating each using macro-averaged F1, per-genre precision/recall, and error analysis.
Dataset
We use the IMDb Genre Classification Dataset from Kaggle (~54,000 movie plot summaries, 27 genre labels).
Place the dataset files in a dataset/ folder in the repo root:

dataset/train_data.txt
dataset/test_data.txt
dataset/test_data_solution.txt

Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn joblib torch nltk
How to Run
The three scripts must be run in order:
Step 1 — Preprocessing (Amulya Dussa)
python preprocessing_nlp.py
Cleans plot summaries, tokenizes, lemmatizes, encodes genre labels, and generates TF-IDF features. Saves all outputs to models/.
Step 2 — Modeling (Maxim Onashchenko)
python models.py
Trains a Logistic Regression baseline and a PyTorch feedforward neural network. Saves trained models to models/.
Step 3 — Evaluation (John Young)
python evaluation.py
Runs per-genre Precision/Recall/F1 analysis, confusion heatmaps, and FP/FN error analysis for both models. Saves all plots to models/.
Results
ModelMicro F1Macro F1Logistic Regression0.49510.1689Neural Network0.60030.2433
Both models struggle with rare genres (war, news, history, musical) due to class imbalance. The neural network outperforms LR particularly in recall for mid-frequency genres.
Notes

Add models/ to .gitignore — it is generated at runtime and contains large files
Neural network training takes approximately 10-15 minutes on CPUes approximately 10–15 minutes.
