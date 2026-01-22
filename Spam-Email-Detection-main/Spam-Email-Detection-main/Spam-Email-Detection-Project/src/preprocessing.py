import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

STOPWORDS = {
    "a","an","the","and","or","but","if","while","with","to","from","of","in","on",
    "for","at","by","is","are","was","were","be","been","this","that","it","as",
    "i","you","he","she","we","they","me","my","your","our","their","so","do","does","did"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df["clean_text"]
    y = df["label"]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )