# TruthLens — Fake News Detection System

TruthLens is an educational machine-learning web application that classifies news text using TF-IDF and Logistic Regression. It includes model evaluation, model comparison, SQLite prediction history, CSV export, and a Streamlit interface.

> **Important:** The model predicts labels learned from a dataset. It does not independently verify whether a claim is true. Predictions can be wrong and should not be treated as definitive fact-checks.

## Features

- News text classification
- TF-IDF feature extraction
- Logistic Regression production model
- Comparison of Logistic Regression, Multinomial Naive Bayes, and Linear SVM
- Accuracy, precision, recall, and F1-score dashboard
- SQLite prediction history
- Search and CSV export of saved predictions
- Automated tests with pytest

## Technology

Python, Pandas, NumPy, Scikit-learn, Streamlit, SQLite, Joblib, Pytest.

## Project structure

```text
TruthLens/
├── app.py
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

## Run the app

```powershell
streamlit run app.py
```

Open http://localhost:8501.

## Run tests

```powershell
python -m pytest -v
```

## GitHub and deployment

Push your source code to GitHub. For Streamlit Community Cloud, make sure the trained model artifact and metadata are available to the deployment. Since the model artifacts are ignored by `.gitignore`, either commit the necessary model files after reviewing their size and licensing implications, or arrange for the deployment to download them from a suitable model-storage location.

Deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Evaluation

The training script evaluates the models on a stratified 20% hold-out split and reports accuracy, precision, recall, F1-score, and confusion matrices. Record your actual measured results here after training; do not claim unmeasured performance.

A random split can overestimate performance if articles share sources, topics, or wording. For stronger evaluation, test on a separate time period or independently sourced dataset.

## Privacy and storage

The app stores submitted article text in a local SQLite database. Do not submit private or sensitive text. On ephemeral hosting, SQLite data may be lost when the app restarts or redeploys. Use persistent storage or a hosted database if durable history is required.

## License

Choose and add a project license before public redistribution. The dataset has its own license, separate from this project's code.
