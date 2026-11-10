import numpy as np
import pytest
import database
import app as web_app
from predictor import predict_news


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, texts):
        return np.array([1])

    def predict_proba(self, texts):
        return np.array([[0.15, 0.85]])


def test_fake_prediction():
    result = predict_news(FakeModel(), "This is a sufficiently long sample article for testing.")
    assert result["prediction"] == "Fake"
    assert result["fake_probability"] == pytest.approx(0.85)
    assert result["real_probability"] == pytest.approx(0.15)


def test_short_article_is_rejected():
    with pytest.raises(ValueError):
        predict_news(FakeModel(), "Too short")


def test_long_article_is_rejected():
    with pytest.raises(ValueError):
        predict_news(FakeModel(), "a" * 50001)


def test_database_history(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.initialize_database()
    database.save_prediction(
        article="A sample news article for database testing.",
        prediction="Real",
        fake_probability=0.20,
        word_count=8,
    )
    history = database.get_history()
    assert len(history) == 1
    assert history.iloc[0]["prediction"] == "Real"
    assert history.iloc[0]["fake_probability"] == pytest.approx(0.20)


def test_clear_history(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_clear.db")
    database.initialize_database()
    database.save_prediction("A sample article for clearing history.", "Fake", 0.90, 7)
    database.clear_history()
    assert database.get_history().empty


def test_web_app_serves_frontend_with_security_headers():
    response = web_app.app.test_client().get("/")

    assert response.status_code == 200
    assert b"TruthLens" in response.data
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_prediction_api_rejects_non_text_input():
    response = web_app.app.test_client().post(
        "/api/predict",
        json={"article": ["not", "text"]},
    )

    assert response.status_code == 400
    assert "text string" in response.get_json()["error"]


def test_prediction_api_treats_injection_like_text_as_article_data(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "web_test.db")
    database.initialize_database()
    monkeypatch.setattr(web_app, "load_model", lambda: FakeModel())
    article = (
        "Officials continue their report. SQL example: ' OR 1=1; "
        "<script>alert('hello')</script>"
    )

    response = web_app.app.test_client().post(
        "/api/predict",
        json={"article": article},
    )

    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Fake"
    history = database.get_history()
    assert len(history) == 1
    assert history.iloc[0]["article"] == article


def test_prediction_api_requires_json():
    response = web_app.app.test_client().post(
        "/api/predict",
        data="article text that is not sent as JSON",
        content_type="text/plain",
    )

    assert response.status_code == 415


def test_prediction_api_enforces_article_character_limit():
    response = web_app.app.test_client().post(
        "/api/predict",
        json={"article": "a" * 50001},
    )

    assert response.status_code == 400
    assert "50,000" in response.get_json()["error"]
