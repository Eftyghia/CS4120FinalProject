import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    multilabel_confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = 'models'
THRESHOLD  = 0.3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load saved artifacts from Preprocessing ─────────────────────────────────────────────
mlb        = joblib.load(f'{MODELS_DIR}/mlb.joblib')
X_val      = joblib.load(f'{MODELS_DIR}/X_val.joblib')
y_val      = joblib.load(f'{MODELS_DIR}/y_val.joblib')
vectorizer = joblib.load(f'{MODELS_DIR}/tfidf_vectorizer.joblib')
genre_names = list(mlb.classes_)
N_CLASSES   = len(genre_names)

# ── Load the LR model and generate predictions ──────────────────────────────
lr_model    = joblib.load(f'{MODELS_DIR}/lr_model.joblib')
lr_probs    = lr_model.predict_proba(X_val)
lr_preds    = (lr_probs >= THRESHOLD).astype(int)

# ── Rebuild the NN architecture and load weights ────────────────────────────
class GenreClassifier(nn.Module):
    def __init__(self, in_features, n_classes, hidden=(512, 256)):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

N_FEATURES = X_val.shape[1]
nn_model   = GenreClassifier(N_FEATURES, N_CLASSES).to(DEVICE)
nn_model.load_state_dict(torch.load(f'{MODELS_DIR}/best_nn.pt',
                                     map_location=DEVICE))
nn_model.eval()

X_val_dense = torch.tensor(X_val.toarray(), dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    nn_probs = torch.sigmoid(nn_model(X_val_dense)).cpu().numpy()
nn_preds = (nn_probs >= THRESHOLD).astype(int)


# ── 1. CLASS WEIGHTING ────────────────────────────────────────────────────────
# Computes how underrepresented each genre is
genre_counts    = y_val.sum(axis=0)
genre_weights   = 1.0 / np.where(genre_counts == 0, 1, genre_counts)
genre_weights  /= genre_weights.sum()

print("=== Class Weights (higher = rarer genre) ===")
for name, w in sorted(zip(genre_names, genre_weights),
                       key=lambda x: -x[1])[:10]:
    print(f"  {name:<20} weight={w:.4f}  count={int(genre_counts[genre_names.index(name)])}")


# ── 2. EACH GENRE'S PRECISION / RECALL / F1 ─────────────────────────────────────
def per_genre_metrics(y_true, y_pred, model_name):
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(N_CLASSES)
    w = 0.25
    ax.bar(x - w, precision, w, label='Precision', color='steelblue')
    ax.bar(x,     recall,    w, label='Recall',    color='coral')
    ax.bar(x + w, f1,        w, label='F1',        color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(genre_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f'Per-Genre Precision / Recall / F1 — {model_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{MODELS_DIR}/per_genre_metrics_{model_name.replace(" ", "_")}.png',
                dpi=150)
    plt.show()
    return precision, recall, f1

print("\n=== LR Per-Genre Metrics ===")
lr_p, lr_r, lr_f1 = per_genre_metrics(y_val, lr_preds, "Logistic Regression")

print("\n=== NN Per-Genre Metrics ===")
nn_p, nn_r, nn_f1 = per_genre_metrics(y_val, nn_preds, "Neural Network")


# ── 3. CONFUSION MATRIX (multilabel, per-genre) ───────────────────────────────
def plot_confusion_heatmap(y_true, y_pred, model_name):
    """
    sklearn's multilabel_confusion_matrix gives a 2x2 matrix per genre.
    We extract TP rate (recall) and FP rate per genre and display as a heatmap.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    tp_rates, fp_rates, fn_rates = [], [], []
    for cm in mcm:
        tn, fp, fn, tp = cm.ravel()
        tp_rates.append(tp / (tp + fn) if (tp + fn) > 0 else 0)  # recall
        fp_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0)  # false positive rate
        fn_rates.append(fn / (fn + tp) if (fn + tp) > 0 else 0)  # miss rate

    summary = np.array([tp_rates, fp_rates, fn_rates])

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(
        summary,
        annot=True, fmt='.2f',
        xticklabels=genre_names,
        yticklabels=['True Positive Rate', 'False Positive Rate', 'False Negative Rate'],
        cmap='RdYlGn', vmin=0, vmax=1, ax=ax
    )
    ax.set_xticklabels(genre_names, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'Per-Genre Confusion Rates — {model_name}')
    plt.tight_layout()
    plt.savefig(f'{MODELS_DIR}/confusion_heatmap_{model_name.replace(" ", "_")}.png',
                dpi=150)
    plt.show()

plot_confusion_heatmap(y_val, lr_preds, "Logistic Regression")
plot_confusion_heatmap(y_val, nn_preds, "Neural Network")


# ── 4. ERROR ANALYSIS ─────────────────────────────────────────────────────────
def error_analysis(y_true, y_pred, model_name):
    print(f"\n=== Error Analysis: {model_name} ===")

    # Most confused genres (highest false positive + false negative counts)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    errors = []
    for i, cm in enumerate(mcm):
        tn, fp, fn, tp = cm.ravel()
        errors.append({
            'genre':  genre_names[i],
            'FP':     int(fp),
            'FN':     int(fn),
            'total_errors': int(fp + fn)
        })

    errors.sort(key=lambda x: -x['total_errors'])

    print(f"{'Genre':<20} {'FP':>6} {'FN':>6} {'Total Errors':>14}")
    print("-" * 50)
    for e in errors:
        print(f"{e['genre']:<20} {e['FP']:>6} {e['FN']:>6} {e['total_errors']:>14}")

    # Bar chart of total errors per genre
    genres_sorted  = [e['genre'] for e in errors]
    fp_sorted      = [e['FP']    for e in errors]
    fn_sorted      = [e['FN']    for e in errors]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(genres_sorted))
    ax.bar(x,     fp_sorted, label='False Positives', color='tomato')
    ax.bar(x, fn_sorted, bottom=fp_sorted, label='False Negatives', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(genres_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'Error Analysis (FP + FN per Genre) — {model_name}')
    ax.set_ylabel('Error Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{MODELS_DIR}/error_analysis_{model_name.replace(" ", "_")}.png',
                dpi=150)
    plt.show()

error_analysis(y_val, lr_preds, "Logistic_Regression")
error_analysis(y_val, nn_preds, "Neural_Network")

print("\nAll plots saved to models/ folder.")