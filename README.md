# TruthLens — Fake News Detection System

TruthLens is an educational machine-learning web application that classifies news text using TF-IDF and Logistic Regression. It includes model evaluation, model comparison, SQLite prediction history, CSV export, and a responsive Flask, HTML, CSS, and JavaScript interface.

> **Important:** The model predicts labels learned from a dataset. It does not independently verify whether a claim is true. Predictions can be wrong and should not be treated as definitive fact-checks.

## Features

- News text classification
- TF-IDF feature extraction
- Logistic Regression production model
- Comparison of Logistic Regression, Multinomial Naive Bayes, and Linear SVM
- Accuracy, precision, recall, and F1-score dashboard
- SQLite prediction history
- Search and CSV export of saved predictions
- Responsive analyze, dashboard, history, and about pages
- Server-side input limits, parameterized SQLite queries, and browser-side text rendering
- Automated tests with pytest

## Technology

Python, Flask, HTML, CSS, JavaScript, Pandas, NumPy, Scikit-learn, SQLite, Joblib, Pytest.

## Project structure

```text
TruthLens/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── css/styles.css
│   └── js/app.js
├── train_model.py
├── predictor.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── (put Fake.csv and True.csv here)
├── models/
│   └── (created by training)
└── tests/
    └── test_app.py
```

## Dataset

Download the Fake and Real News Dataset from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Extract `Fake.csv` and `True.csv` into the `data/` folder. Review the dataset's license and attribution requirements before redistribution or commercial use. This project does not bundle the dataset.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train the models

```powershell
python train_model.py
```

Training creates `models/news_model.joblib`, `models/metrics.json`, `models/model_comparison.csv`, and `models/misclassified_examples.csv`.

The app expects a trained model. The model files are not included in this starter ZIP because they are generated from the dataset.

## Run the local web app

```powershell
python app.py
```

Open http://127.0.0.1:5000.

The app binds to localhost and runs with Flask debug mode disabled. Article input is checked on the server for type and length; SQL statements use bound parameters, and article/history text is rendered as text rather than executable markup. Security checks intentionally do not reject SQL-looking words or HTML-like text, since those can be part of a news article.

## Run tests

```powershell
python -m pytest -v
```

## GitHub and deployment

For deployment, use a production WSGI server rather than Flask's built-in development server. Make the trained model artifact and metadata available to the deployment; model artifacts are ignored by `.gitignore`, so arrange suitable model storage and review licensing and privacy requirements.

## Evaluation

The training script evaluates the models on a stratified 20% hold-out split and reports accuracy, precision, recall, F1-score, and confusion matrices. Record your actual measured results here after training; do not claim unmeasured performance.

A random split can overestimate performance if articles share sources, topics, or wording. For stronger evaluation, test on a separate time period or independently sourced dataset.

## Privacy and storage

The app stores submitted article text in a local SQLite database. Do not submit private or sensitive text. On ephemeral hosting, SQLite data may be lost when the app restarts or redeploys. Use persistent storage or a hosted database if durable history is required.

## License

Choose and add a project license before public redistribution. The dataset has its own license, separate from this project's code.
