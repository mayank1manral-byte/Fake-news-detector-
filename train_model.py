from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset():
    fake_file = DATA_DIR / "Fake.csv"
    real_file = DATA_DIR / "True.csv"
    if not fake_file.exists() or not real_file.exists():
        raise FileNotFoundError(
            "Download the Kaggle dataset and place Fake.csv and True.csv in data/."
        )

    frames = []
    for path, label in [(fake_file, 1), (real_file, 0)]:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        if "text" not in df.columns:
            raise ValueError(f"{path.name} must contain a 'text' column.")
        if "title" not in df.columns:
            df["title"] = ""
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)
        df["content"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()
        df["label"] = label
        frames.append(df[["content", "label"]])

    data = pd.concat(frames, ignore_index=True)
    data = data[data["content"].str.len() > 0]
    data = data.drop_duplicates(subset=["content"])
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def create_models():
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }
    return {
        name: Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2),
                max_features=100000, min_df=2, sublinear_tf=True
            )),
            ("classifier", classifier),
        ])
        for name, classifier in classifiers.items()
    }


def main():
    data = load_dataset()
    X, y = data["content"], data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    trained = {}
    for name, pipeline in create_models().items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        }
        results.append(row)
        trained[name] = pipeline
        print(row)
        print(classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Real", "Fake"], zero_division=0
        ))
        print("Confusion matrix (rows=actual, columns=predicted; Real, Fake):")
        print(confusion_matrix(y_test, pred, labels=[0, 1]))

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    production = trained["Logistic Regression"]
    joblib.dump(production, MODEL_DIR / "news_model.joblib")
    lr_metrics = next(r for r in results if r["model"] == "Logistic Regression")

    metadata = {
        "production_model": "Logistic Regression",
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "test_size": 0.20,
        "label_mapping": {"0": "Real", "1": "Fake"},
        "production_metrics": lr_metrics,
        "model_comparison": results,
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    errors = pd.DataFrame({
        "text": X_test.reset_index(drop=True),
        "actual": y_test.reset_index(drop=True),
        "predicted": production.predict(X_test),
    })
    errors = errors[errors["actual"] != errors["predicted"]]
    errors.to_csv(MODEL_DIR / "misclassified_examples.csv", index=False)

    print("\nTraining completed.")
    print(f"Saved model to: {MODEL_DIR / 'news_model.joblib'}")


if __name__ == "__main__":
    main()
