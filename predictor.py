from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "news_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Download the dataset and run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_news(model, article):
    article = article.strip()
    if len(article) < 30:
        raise ValueError("Please enter at least 30 characters of news text.")
    if len(article) > 50000:
        raise ValueError("Please keep the news text below 50,000 characters.")

    prediction = int(model.predict([article])[0])
    probabilities = model.predict_proba([article])[0]
    classes = list(model.classes_)
    fake_probability = float(probabilities[classes.index(1)])
    real_probability = float(probabilities[classes.index(0)])

    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "word_count": len(article.split()),
    }
