import argparse
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = 'models'
DATA_DIR = 'dataset'
BERT_DIR = os.path.join(MODELS_DIR, 'bert')

# Switch to "bert-base-uncased" if you have a powerful GPU and want full BERT.
# distilbert is ~40 % faster with minimal accuracy drop.
MODEL_NAME = 'distilbert-base-uncased'

MAX_LEN = 256  # plot summaries are long; 256 captures most content
BATCH_SIZE = 16  # reduce to 8 if you hit OOM
EPOCHS = 4
LR = 2e-5
WARMUP_RATIO = 0.1
THRESHOLD = 0.3  # same threshold used by the error-analysis script

os.makedirs(BERT_DIR, exist_ok=True)

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu')
)
print(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class PlotDataset(Dataset):
    """
    Tokenises raw plot text on the fly so we do NOT depend on TF-IDF
    features. BERT needs the original text, not the cleaned version.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels  # numpy array (n_samples, n_classes)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Load preprocessed artifacts ──────────────────────────────────────────────
def load_artifacts():
    mlb       = joblib.load(os.path.join(MODELS_DIR, 'mlb.joblib'))
    train_df  = joblib.load(os.path.join(MODELS_DIR, 'train_processed.pkl'))
    test_sol  = joblib.load(os.path.join(MODELS_DIR, 'test_solution_processed.pkl'))

    print(f"train_processed columns:         {list(train_df.columns)}")
    print(f"test_solution_processed columns: {list(test_sol.columns)}")

    def make_texts(df):
        if 'text' in df.columns:
            return df['text'].fillna('').tolist()
        elif 'clean_plot' in df.columns:
            return (df['title'].fillna('') + ' ' + df['clean_plot'].fillna('')).tolist()
        elif 'plot' in df.columns:
            return (df['title'].fillna('') + ' ' + df['plot'].fillna('')).tolist()
        else:
            raise ValueError(f"Cannot find text column. Available: {list(df.columns)}")

    train_texts = make_texts(train_df)
    test_texts  = make_texts(test_sol)

    y_train = mlb.transform(train_df['genres'])
    y_test  = mlb.transform(test_sol['genres'])

    return mlb, train_texts, y_train, test_texts, y_test


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='  train', leave=False):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        # AutoModelForSequenceClassification returns logits in .logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, threshold=THRESHOLD):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    preds = (all_probs >= threshold).astype(int)

    micro_f1 = f1_score(all_labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)

    return total_loss / len(loader), micro_f1, macro_f1, all_probs, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main(eval_only=False):
    mlb, train_texts, y_train, test_texts, y_test = load_artifacts()
    n_classes = len(mlb.classes_)
    print(f"Classes: {n_classes} | Train samples: {len(train_texts):,} | Test samples: {len(test_texts):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Split off a validation set (same 85/15 split as the other models)
    from sklearn.model_selection import train_test_split
    tr_texts, val_texts, y_tr, y_val = train_test_split(
        train_texts, y_train, test_size=0.15, random_state=42
    )

    train_ds = PlotDataset(tr_texts, y_tr, tokenizer, MAX_LEN)
    val_ds = PlotDataset(val_texts, y_val, tokenizer, MAX_LEN)
    test_ds = PlotDataset(test_texts, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Build model — AutoModelForSequenceClassification sets the head to
    # n_classes outputs; we apply BCEWithLogitsLoss ourselves for multi-label.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        problem_type='multi_label_classification',
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(BERT_DIR, 'best_bert.pt')

    if eval_only:
        print(f"\nLoading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        print("\nFine-tuning BERT...")
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            tr_loss = train(model, train_loader, optimizer, scheduler, criterion)
            val_loss, val_micro, val_macro, _, _ = evaluate(model, val_loader, criterion)

            print(f"  Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Micro F1: {val_micro:.4f} | Val Macro F1: {val_macro:.4f}")

            if val_micro > best_val_f1:
                best_val_f1 = val_micro
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (Val Micro F1={best_val_f1:.4f})")

        print(f"\nBest Val Micro F1: {best_val_f1:.4f}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # ── Final evaluation on the held-out test set ─────────────────────────
    print("\n=== Test Set Evaluation ===")
    _, test_micro, test_macro, test_probs, test_labels = evaluate(
        model, test_loader, criterion
    )
    test_preds = (test_probs >= THRESHOLD).astype(int)

    print(f"Test Micro F1: {test_micro:.4f}")
    print(f"Test Macro F1: {test_macro:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=mlb.classes_,
        zero_division=0,
    ))

    # Save predictions so the error-analysis script can load them
    joblib.dump(test_probs, os.path.join(BERT_DIR, 'bert_test_probs.joblib'))
    joblib.dump(test_preds, os.path.join(BERT_DIR, 'bert_test_preds.joblib'))
    joblib.dump(test_labels, os.path.join(BERT_DIR, 'bert_test_labels.joblib'))
    print(f"\nPredictions saved to {BERT_DIR}/")

    # ── Summary table (mirrors the bar chart in the MLP script) ──────────
    # Load the saved LR and MLP scores if available so you can print a
    # clean comparison without re-running those models.
    try:
        lr_model = joblib.load(os.path.join(MODELS_DIR, 'lr_model.joblib'))
        X_val_tfidf = joblib.load(os.path.join(MODELS_DIR, 'X_val.joblib'))
        y_val_saved = joblib.load(os.path.join(MODELS_DIR, 'y_val.joblib'))
        lr_preds = (lr_model.predict_proba(X_val_tfidf) >= THRESHOLD).astype(int)
        lr_micro = f1_score(y_val_saved, lr_preds, average='micro', zero_division=0)
        lr_macro = f1_score(y_val_saved, lr_preds, average='macro', zero_division=0)

        _, val_micro_nn, val_macro_nn, _, _ = (None, None, None, None, None)  # placeholder
        # Try loading NN val preds if they were saved separately
        print("\n=== Model Comparison (val set) ===")
        print(f"{'Model':<25} {'Micro F1':>10} {'Macro F1':>10}")
        print("-" * 47)
        print(f"{'Logistic Regression':<25} {lr_micro:>10.4f} {lr_macro:>10.4f}")
        print(f"{'BERT (test set)':<25} {test_micro:>10.4f} {test_macro:>10.4f}")
        print("\nNote: run model_training.py to get the MLP row.")
    except FileNotFoundError:
        pass  # baselines not yet saved, that's fine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training and evaluate saved weights')
    args = parser.parse_args()
    main(eval_only=args.eval_only)