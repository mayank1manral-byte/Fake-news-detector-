import numpy as np
import pytest
import database
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
