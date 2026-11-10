import io
import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from database import clear_history, get_history, initialize_database, save_prediction
from predictor import load_model, predict_news

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
MAX_ARTICLE_LENGTH = 50000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 320 * 1024


@app.after_request
def add_security_headers(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    return jsonify(error="Request is too large. Keep the news text below 50,000 characters."), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def analyze_news():
    if not request.is_json:
        return jsonify(error="Send the article as JSON with an 'article' string."), 415

    try:
        payload = request.get_json()
    except BadRequest:
        return jsonify(error="The request body must contain valid JSON."), 400

    if not isinstance(payload, dict) or not isinstance(payload.get("article"), str):
        return jsonify(error="The article must be a text string."), 400

    article = payload["article"].strip()
    if len(article) < 30:
        return jsonify(error="Please enter at least 30 characters of news text."), 400
    if len(article) > MAX_ARTICLE_LENGTH:
        return jsonify(error="Please keep the news text below 50,000 characters."), 400

    try:
        result = predict_news(load_model(), article)
    except FileNotFoundError as error:
        return jsonify(error=str(error)), 503
    except ValueError as error:
        return jsonify(error=str(error)), 400

    save_prediction(
        article,
        result["prediction"],
        result["fake_probability"],
        result["word_count"],
    )
    result["character_count"] = len(article)
    return jsonify(result)


@app.get("/api/metrics")
def model_metrics():
    metrics = None
    comparison = []
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH).to_dict(orient="records")
    return jsonify(metrics=metrics, comparison=comparison)


@app.get("/api/history")
def prediction_history():
    history = get_history(limit=500)
    query = request.args.get("q", "").strip().casefold()
    if query:
        history = history[
            history["article"].str.casefold().str.contains(query, regex=False, na=False)
        ]
    return jsonify(records=history.to_dict(orient="records"))


@app.get("/api/history.csv")
def export_history():
    history = get_history(limit=500)
    query = request.args.get("q", "").strip().casefold()
    if query:
        history = history[
            history["article"].str.casefold().str.contains(query, regex=False, na=False)
        ]
    csv_file = io.BytesIO(history.to_csv(index=False).encode("utf-8"))
    return send_file(
        csv_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="truthlens_history.csv",
    )


@app.delete("/api/history")
def delete_history():
    clear_history()
    return jsonify(message="Prediction history cleared.")


@app.errorhandler(404)
def handle_not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="API endpoint not found."), 404
    return "Page not found.", 404


@app.errorhandler(405)
def handle_method_not_allowed(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="Method not allowed."), 405
    return "Method not allowed.", 405


initialize_database()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
