import numpy as np
import joblib
import torch
import sys
sys.path.append('.')
from preprocessing_nlp import load_data, clean_text
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

MODELS_DIR = 'models'
DATA_DIR   = 'data/Genre Classification Dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mlb       = joblib.load(f'{MODELS_DIR}/mlb.joblib')
X_train   = joblib.load(f'{MODELS_DIR}/X_train.joblib')
X_val     = joblib.load(f'{MODELS_DIR}/X_val.joblib')
y_train   = joblib.load(f'{MODELS_DIR}/y_train.joblib')
y_val     = joblib.load(f'{MODELS_DIR}/y_val.joblib')
X_test    = joblib.load(f'{MODELS_DIR}/X_test_tfidf.joblib')
vectorizer = joblib.load(f'{MODELS_DIR}/tfidf_vectorizer.joblib')


def predict_genres(plot_text, title="", threshold=None):
    """Run inference on a raw plot string using imported helpers."""
    thresh = threshold or best_thresh
    text = title + " " + clean_text(plot_text)
    vec = vectorizer.transform([text])
    X_t = torch.tensor(vec.toarray(), dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    return mlb.inverse_transform((probs >= thresh).astype(int))[0]


def load_raw(split='test'):
    """Reload raw data using the imported load_data helper."""
    has_genre = split != 'test'
    path = f'{DATA_DIR}/{"test_data" if split == "test" else split + "_data"}.txt'
    df = load_data(path, has_genre=has_genre)
    df['clean_plot'] = df['plot'].apply(clean_text)
    df['text'] = df['title'] + " " + df['clean_plot']
    return df

N_CLASSES  = len(mlb.classes_)
N_FEATURES = X_train.shape[1]
print(f"Features: {N_FEATURES} | Classes: {N_CLASSES}")


print("LR")

lr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
    n_jobs=-1
)
lr_model.fit(X_train, y_train)

lr_val_probs = lr_model.predict_proba(X_val)
lr_val_preds = (lr_val_probs >= 0.5).astype(int)

lr_f1_micro = f1_score(y_val, lr_val_preds, average='micro', zero_division=0)
lr_f1_macro = f1_score(y_val, lr_val_preds, average='macro', zero_division=0)

print(f"Validation Micro F1: {lr_f1_micro:.4f}")
print(f"Validation Macro F1: {lr_f1_macro:.4f}")
print("Per-class report:")
print(classification_report(y_val, lr_val_preds,
                             target_names=mlb.classes_, zero_division=0))

joblib.dump(lr_model, f'{MODELS_DIR}/lr_model.joblib')


def to_tensor(X, y=None):
    X_t = torch.tensor(X.toarray(), dtype=torch.float32)
    if y is not None:
        return TensorDataset(X_t, torch.tensor(y, dtype=torch.float32))
    return X_t

train_ds = to_tensor(X_train, y_train)
val_ds   = to_tensor(X_val, y_val)
X_test_t = to_tensor(X_test)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)


class GenreClassifier(nn.Module):
    def __init__(self, in_features, n_classes, hidden=(512, 256)):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model     = GenreClassifier(N_FEATURES, N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_b)
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(y_b.cpu().numpy())
    return total_loss / len(loader.dataset), np.vstack(all_preds), np.vstack(all_labels)


def threshold_f1(probs, labels, thresholds=np.arange(0.2, 0.6, 0.05)):
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(labels, (probs >= t).astype(int), average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


print("neural network training")

EPOCHS  = 20
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_f1, best_thresh = 0.0, 0.5

for epoch in range(1, EPOCHS + 1):
    tr_loss, _, _                   = run_epoch(train_loader, train=True)
    val_loss, val_probs, val_labels = run_epoch(val_loader,   train=False)
    thresh, val_f1                  = threshold_f1(val_probs, val_labels)
    scheduler.step(val_f1)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {tr_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Micro F1: {val_f1:.4f} thresh={thresh:.2f}")

    if val_f1 > best_f1:
        best_f1, best_thresh = val_f1, thresh
        torch.save(model.state_dict(), f'{MODELS_DIR}/best_nn.pt')
        print(f"best F1={best_f1:.4f}")


model.load_state_dict(torch.load(f'{MODELS_DIR}/best_nn.pt'))

_, val_probs, val_labels = run_epoch(val_loader, train=False)
val_preds = (val_probs >= best_thresh).astype(int)

print(f"Final Val Micro F1: {f1_score(val_labels, val_preds, average='micro', zero_division=0):.4f}")
print(f"Final Val Macro F1: {f1_score(val_labels, val_preds, average='macro', zero_division=0):.4f}")
print(classification_report(val_labels, val_preds, target_names=mlb.classes_, zero_division=0))

model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(X_test_t.to(DEVICE))).cpu().numpy()

test_genres = mlb.inverse_transform((test_probs >= best_thresh).astype(int))
joblib.dump(test_genres, f'{MODELS_DIR}/nn_test_predictions.joblib')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'],   label='Val Loss')
axes[0].set_title('Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].legend()

models_names = ['LR Baseline', 'Neural Network']
f1_scores    = [lr_f1_micro, f1_score(val_labels, val_preds, average='micro', zero_division=0)]
axes[1].bar(models_names, f1_scores, color=['steelblue', 'coral'])
axes[1].set_ylim(0, 1)
axes[1].set_title('Micro F1 — Validation Set')
axes[1].set_ylabel('F1 Score')
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{MODELS_DIR}/model_comparison.png', dpi=150)
plt.show()