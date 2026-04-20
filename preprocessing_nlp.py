"""
Natural Language Processing
IMDb Genre Classification Project

Author: Amulya Dussa
April 16, 2026
"""
import pandas as pd
import re
import nltk
import os
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_DIR = 'dataset'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)


# Loading Data
def load_data(file_path, has_genre=True):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(' ::: ')]

            if has_genre and len(parts) >= 4:
                data.append({
                    'id': parts[0],
                    'title': parts[1],
                    'genres': parts[2].split(),
                    'plot': parts[3]
                })
            elif not has_genre and len(parts) >= 3:
                data.append({
                    'id': parts[0],
                    'title': parts[1],
                    'plot': parts[2]
                })
    return pd.DataFrame(data)


print("Loading datasets") # sanity check to make sure its loading
train_df = load_data(f'{DATA_DIR}/train_data.txt', has_genre=True)
test_df = load_data(f'{DATA_DIR}/test_data.txt', has_genre=False)
test_solution_df = load_data(f'{DATA_DIR}/test_data_solution.txt', has_genre=True)

print(f"Train samples: {len(train_df):,}")
print(f"Test samples:  {len(test_df):,}")

# Text cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


print("Cleaning plot summaries")
train_df['clean_plot'] = train_df['plot'].apply(clean_text)
test_df['clean_plot'] = test_df['plot'].apply(clean_text)
test_solution_df['clean_plot'] = test_solution_df['plot'].apply(clean_text)

train_df['text'] = train_df['title'] + " " + train_df['clean_plot']
test_df['text'] = test_df['title'] + " " + test_df['clean_plot']

# Multi label and TF - IDF
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df['genres'])

print(f"\nNumber of genres: {len(mlb.classes_)}")
print("Genres:", mlb.classes_)

print("\nCreating TF-IDF features:")
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85
)

X_tfidf = vectorizer.fit_transform(train_df['text'])
X_test_tfidf = vectorizer.transform(test_df['text'])

print(f"TF-IDF shape: {X_tfidf.shape}")

# Train Value Split
X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, y, test_size=0.15, random_state=42
)

# Save with Joblib
print("\nSaving files...")

joblib.dump(mlb, f'{MODELS_DIR}/mlb.joblib')
joblib.dump(vectorizer, f'{MODELS_DIR}/tfidf_vectorizer.joblib')

joblib.dump(X_train, f'{MODELS_DIR}/X_train.joblib')
joblib.dump(X_val, f'{MODELS_DIR}/X_val.joblib')
joblib.dump(y_train, f'{MODELS_DIR}/y_train.joblib')
joblib.dump(y_val, f'{MODELS_DIR}/y_val.joblib')
joblib.dump(X_test_tfidf, f'{MODELS_DIR}/X_test_tfidf.joblib')

# Save dataframes
train_df[['id', 'title', 'text', 'genres']].to_pickle(f'{MODELS_DIR}/train_processed.pkl')
test_df[['id', 'title', 'text']].to_pickle(f'{MODELS_DIR}/test_processed.pkl')
test_solution_df.to_pickle(f'{MODELS_DIR}/test_solution_processed.pkl')

print(f"All files saved in '{MODELS_DIR}' folder")

# Genre Distribution Plot
plt.figure(figsize=(14, 7))
all_genres = [g for sublist in train_df['genres'] for g in sublist]
genre_counts = pd.Series(all_genres).value_counts().head(20)

sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.title('Top 20 Most Common Genres')
plt.xlabel('Number of Movies')
plt.tight_layout()
plt.show()