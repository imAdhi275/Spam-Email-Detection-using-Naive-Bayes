import os
import pandas as pd

from src.preprocessing import load_dataset, split_data, clean_text
from src.train_model import (
    get_vectorizers, vectorize,
    train_multinomial_nb, train_gaussian_nb
)
from src.evaluate import evaluate_model
from src.predict import predict_email

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = load_dataset("dataset/Spam-Email-Detection.csv")
X_train, X_test, y_train, y_test = split_data(df)

bow, tfidf = get_vectorizers()
results = []

for name, vectorizer in {"BoW": bow, "TFIDF": tfidf}.items():
    X_train_vec, X_test_vec = vectorize(vectorizer, X_train, X_test)

    mnb = train_multinomial_nb(X_train_vec, y_train)
    results.append(
        evaluate_model(
            mnb, "multinomial", X_test_vec, y_test,
            f"{name}_MultinomialNB", OUTPUT_DIR
        )
    )

    gnb = train_gaussian_nb(X_train_vec, y_train)
    results.append(
        evaluate_model(
            gnb, "gaussian", X_test_vec, y_test,
            f"{name}_GaussianNB", OUTPUT_DIR
        )
    )

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

print("\nFINAL COMPARISON TABLE")
print(results_df)

# Real-world demo
best_vectorizer = tfidf
X_all = best_vectorizer.fit_transform(df["clean_text"])
best_model = train_multinomial_nb(X_all, df["label"])

emails = [
    "Congratulations you won a free prize",
    "Sir I have submitted the assignment",
    "Urgent bank account blocked",
    "Are we meeting tomorrow?"
]

print("\nREAL WORLD EMAIL TESTING")
for email in emails:
    label, prob = predict_email(
        email, clean_text, best_vectorizer, best_model
    )
    print(email)
    print("Prediction:", label, "| Spam Probability:", prob)