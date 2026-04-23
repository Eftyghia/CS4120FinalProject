# CS4120FinalProject — IMDb Multi-Label Genre Classification
Course: CS4120 Natural Language Processing
Group Members: Amulya Dussa, Eftyghia Kourtelidis, Maxim Onashchenko, John Young


## Project Overview
This project tackles multi-label genre classification of movies using NLP techniques applied to IMDb plot summaries. Given a textual description of a film, our system predicts one or more genre labels from a set of 27 categories (e.g. Drama, Comedy, Horror, Thriller). We build and compare a TF-IDF + Logistic Regression baseline, a custom feedforward neural network, and a fine-tuned BERT transformer (in progress), evaluating each using macro-averaged F1, per-genre precision/recall, and error analysis.


## Dataset
We use the IMDb Genre Classification Dataset from Kaggle (~54,000 movie plot summaries, 27 genre labels). The label distribution is heavily skewed — Drama dominates with ~13,000 instances while genres like Game-Show and News appear in fewer than 1% of entries.


## Place the dataset files in a dataset/ folder in the repo root:
dataset/train_data.txt
dataset/test_data.txt
dataset/test_data_solution.txt


## Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn joblib torch nltk transformers tqdm
NLTK data packages will download automatically on first run.


## How to Run
The four scripts must be run in order. Each script depends on the outputs of the previous one.


### Step 1 — Preprocessing (Amulya Dussa)

python preprocessing_nlp.py

Loads raw plot summaries and cleans them via lowercasing, punctuation removal, tokenization, stopword filtering, and lemmatization using NLTK. Encodes genre labels as 27-dimensional binary vectors using scikit-learn's MultiLabelBinarizer. Generates TF-IDF feature matrices with 8,000 features using unigrams and bigrams. Saves all processed data and fitted transformers to models/ as .joblib files.

Outputs saved to models/:
mlb.joblib — fitted MultiLabelBinarizer
tfidf_vectorizer.joblib — fitted TF-IDF vectorizer
X_train.joblib, X_val.joblib, y_train.joblib, y_val.joblib — train/val splits
X_test_tfidf.joblib — test features
train_processed.pkl, test_processed.pkl, test_solution_processed.pkl — processed dataframes


### Step 2 — Modeling (Maxim Onashchenko)

python models.py

Trains two models on the preprocessed TF-IDF features. First, a Logistic Regression baseline using a One-vs-Rest strategy with scikit-learn. Second, a custom PyTorch feedforward neural network with two hidden layers (512 and 256 units), ReLU activations, BatchNorm, and Dropout. Both models use a sigmoid output layer for multi-label prediction and optimize binary cross-entropy loss. Decision thresholds are tuned on the validation set by sweeping values between 0.20 and 0.55 to maximize micro F1. Trains for 20 epochs with the Adam optimizer and a ReduceLROnPlateau scheduler.

Outputs saved to models/:

lr_model.joblib — trained Logistic Regression model
best_nn.pt — best neural network weights
model_comparison.png — loss curves and micro F1 comparison chart


### Step 3 — Evaluation (John Young)
python evaluation.py

Loads both trained models and runs full evaluation on the validation set. Computes class weights to quantify genre imbalance across all 27 genres. Generates per-genre Precision, Recall, and F1 bar charts for both models. Produces per-genre confusion heatmaps showing true positive, false positive, and false negative rates. Runs a detailed FP/FN error analysis ranking genres by total error count. All plots are saved to models/ and a ranked error analysis table is printed to the terminal.

Outputs saved to models/:

per_genre_metrics_Logistic_Regression.png
per_genre_metrics_Neural_Network.png
confusion_heatmap_Logistic_Regression.png
confusion_heatmap_Neural_Network.png
error_analysis_Logistic_Regression.png
error_analysis_Neural_Network.png


## Step 4 - BERT Classification (Eftyghia Kourtelidis)

python bert_classification.py

Fine-tunes a DistilBERT transformer model (distilbert-base-uncased) for multi-label genre classification. Unlike the TF-IDF models, DistilBERT receives raw unprocessed plot text and title directly, using the HuggingFace AutoTokenizer with max_length=256. A linear classification head is added on top of the encoder with BCEWithLogitsLoss for multi-label output. Fine-tuned for 4 epochs using AdamW with a learning rate of 2e-5, weight decay of 0.01, and a linear warmup schedule. Genre predictions are produced by thresholding sigmoid probabilities at 0.3. Evaluates on the held-out test set and saves predictions to models/bert/.

To skip training and evaluate using saved weights: python bert_classification.py --eval-only

Outputs saved to models/bert/:

best_bert.pt — best model weights
bert_test_probs.joblib — predicted probabilities on test set
bert_test_preds.joblib — binary predictions on test set
bert_test_labels.joblib — ground truth labels on test set

Note: GPU is strongly recommended for BERT fine-tuning. The script will automatically use CUDA if available, then MPS (Apple Silicon), then CPU.

## Results

Models:
Logistic Regression: Micro F1 = 0.4951, Macro F1 = 0.1689, Eval Set = Validation
Neural Network: Micro F1 = 0.6003, Macro F1 = 0.2433, Eval Set = Validation
DistilBERT: Micro F1 = 0.6667, Macro F1 = 0.4265, Eval Set = Test

Each model improves meaningfully over the last. The Macro F1 jump from Neural Network (0.2433) to DistilBERT (0.4265) is the most significant gain, reflecting DistilBERT's substantially better handling of rare genres. DistilBERT is the first model to produce non-zero F1 for genres including Animation, Family, Romance, Sci-Fi, Talk-Show, and Sport. Biography, History, and Mystery remain near-zero across all three models due to severe class imbalance and thematic overlap with more common genres.


## Notes
Add models/ to .gitignore — it is generated at runtime and contains large files
Neural network training takes approximately 10-15 minutes on CPUes approximately 10–15 minutes.
LR and NN results are evaluated on the validation set; DistilBERT is evaluated on the held-out test set
The models.py script imports from preprocessing_nlp.py — run preprocessing first to avoid errors
