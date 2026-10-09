import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "truthlens.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%d %H:%M:%S', 'now')),
                article TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fake_probability REAL NOT NULL,
                word_count INTEGER NOT NULL
            )
        """)


def save_prediction(article, prediction, fake_probability, word_count):
    with get_connection() as connection:
        connection.execute(
            """INSERT INTO predictions
               (article, prediction, fake_probability, word_count)
               VALUES (?, ?, ?, ?)""",
            (article, prediction, float(fake_probability), int(word_count)),
        )


def get_history(limit=200):
    with get_connection() as connection:
        return pd.read_sql_query(
            """SELECT id, created_at, article, prediction,
                      fake_probability, word_count
               FROM predictions ORDER BY id DESC LIMIT ?""",
            connection,
            params=(int(limit),),
        )


def clear_history():
    with get_connection() as connection:
        connection.execute("DELETE FROM predictions")
