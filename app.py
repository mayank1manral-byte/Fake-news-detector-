import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from database import clear_history, get_history, initialize_database, save_prediction
from predictor import load_model, predict_news

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="TruthLens | News Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #eaf0f6;
        --muted: #91a0b2;
        --panel: #111b27;
        --panel-strong: #152231;
        --line: rgba(174, 196, 218, .13);
        --teal: #4ed9c1;
        --blue: #79aaff;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 78% 0%, rgba(32, 111, 121, .15), transparent 31%),
            #0b121b;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(11, 18, 27, .72); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container {
        max-width: 1320px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #0d1722;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.7rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .62rem .8rem;
        border-radius: 10px;
        transition: background .15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, .055);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        color: #bdc9d6 !important;
        font-weight: 600;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Segoe UI', sans-serif !important;
        letter-spacing: -.035em;
    }
    h1 { font-size: clamp(2.15rem, 4vw, 3.45rem) !important; line-height: 1.08 !important; }
    h2 { font-size: 1.5rem !important; }
    p, li { color: #bdc9d6; }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: 0 0 1.55rem .15rem;
    }
    .brand-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        color: #09201f;
        background: linear-gradient(145deg, #6ce8d1, #35bdaa);
        border-radius: 13px;
        font-size: 1.15rem;
        font-weight: 800;
        box-shadow: 0 8px 22px rgba(46, 199, 177, .18);
    }
    .brand-name { color: #f1f6fa; font: 800 1.18rem 'Segoe UI', sans-serif; }
    .brand-caption { color: #8292a4; font-size: .76rem; margin-top: .1rem; }
    .eyebrow {
        color: var(--teal);
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .75rem;
    }
    .page-intro { max-width: 760px; margin: 0 0 2rem; }
    .page-intro h1 { color: #f3f7fb; margin: 0 0 .72rem; }
    .page-intro p { color: #9cabb9; font-size: 1.02rem; line-height: 1.7; margin: 0; }
    .surface {
        background: linear-gradient(150deg, rgba(21, 34, 49, .94), rgba(15, 25, 37, .96));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 16px 44px rgba(0, 0, 0, .12);
    }
    .surface-label {
        color: #8fa1b3;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 0 0 .6rem;
    }
    .surface h3 { color: #edf4f9; font-size: 1.12rem !important; margin: 0 0 .5rem; }
    .surface p { color: #9eacba; font-size: .91rem; line-height: 1.65; margin: 0; }
    .step-row {
        display: flex;
        gap: .85rem;
        align-items: flex-start;
        margin: 1.15rem 0;
    }
    .step-number {
        display: grid;
        place-items: center;
        flex: 0 0 27px;
        height: 27px;
        border: 1px solid rgba(78, 217, 193, .3);
        border-radius: 9px;
        color: var(--teal);
        font-size: .76rem;
        font-weight: 700;
    }
    .step-copy { color: #a9b7c5; font-size: .87rem; line-height: 1.5; }
    .step-copy strong { color: #e4edf5; display: block; margin-bottom: .1rem; }
    .result-panel {
        border-radius: 16px;
        border: 1px solid var(--line);
        padding: 1.2rem 1.35rem;
        margin: 1rem 0;
        background: rgba(17, 27, 39, .92);
    }
    .result-panel.fake { border-color: rgba(255, 117, 117, .35); }
    .result-panel.real { border-color: rgba(78, 217, 193, .35); }
    .result-kicker {
        color: #93a3b4;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .result-title { color: #f2f6fa; font: 800 1.55rem 'Segoe UI', sans-serif; }
    .result-panel.fake .result-title { color: #ff9696; }
    .result-panel.real .result-title { color: var(--teal); }
    [data-testid="stTextArea"] textarea {
        background: #0c1621;
        border: 1px solid rgba(174, 196, 218, .18);
        border-radius: 12px;
        color: #eaf0f6;
        line-height: 1.6;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(78, 217, 193, .7);
        box-shadow: 0 0 0 1px rgba(78, 217, 193, .2);
    }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #0c1621;
        border-color: rgba(174, 196, 218, .18);
        border-radius: 10px;
        color: var(--ink);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        border: 1px solid rgba(78, 217, 193, .42);
        border-radius: 10px;
        background: linear-gradient(135deg, #4ed9c1, #37bfae);
        color: #09201f;
        font-weight: 700;
        min-height: 2.75rem;
        transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        color: #071a19;
        filter: brightness(1.07);
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background: rgba(17, 27, 39, .9);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: #97a7b7; }
    [data-testid="stMetricValue"] { color: #f2f6fa; }
    [data-testid="stProgressBar"] > div > div { background: var(--teal); }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: var(--line); }
    .fine-print { color: #78899a; font-size: .78rem; line-height: 1.55; }
    @media (max-width: 700px) {
        .block-container { padding: 1.5rem 1rem 3rem; }
        .surface { padding: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_comparison():
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


def render_page_intro(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="page-intro">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


initialize_database()
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-icon">T</div>
            <div>
                <div class="brand-name">TruthLens</div>
                <div class="brand-caption">News analysis studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Explore the signals behind a news story.")
    page = st.radio(
        "Workspace",
        ["Analyze News", "Dashboard", "History", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        '<p class="fine-print">A machine-learning estimate is not proof of truth or falsity. Verify important claims independently.</p>',
        unsafe_allow_html=True,
    )


if page == "Analyze News":
    render_page_intro(
        "News credibility • Text analysis",
        "Read beyond the headline.",
        "Explore how a trained language model classifies a news article. Paste a headline or article to get an instant, explainable probability breakdown.",
    )

    input_column, guide_column = st.columns([1.65, 1], gap="large")
    with input_column:
        st.markdown('<div class="surface-label">01 / Article input</div>', unsafe_allow_html=True)
        st.markdown("### What would you like to analyze?")
        with st.form("news_form"):
            article = st.text_area(
                "News article",
                height=250,
                max_chars=50000,
                placeholder="Paste the headline and article text here...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "Analyze article  →", type="primary", use_container_width=True
            )
        st.markdown(
            '<p class="fine-print">For the most useful result, include the complete article rather than only a headline. Text is stored in local prediction history.</p>',
            unsafe_allow_html=True,
        )

    with guide_column:
        st.markdown(
            """
            <div class="surface">
                <div class="surface-label">How it works</div>
                <h3>Text patterns, not fact-checking</h3>
                <p>TruthLens compares the writing patterns in your text with examples the model learned from its training dataset.</p>
                <div class="step-row">
                    <div class="step-number">1</div>
                    <div class="step-copy"><strong>Text features</strong>TF-IDF represents words and phrases as model features.</div>
                </div>
                <div class="step-row">
                    <div class="step-number">2</div>
                    <div class="step-copy"><strong>Classification</strong>Logistic Regression estimates the dataset label.</div>
                </div>
                <div class="step-row">
                    <div class="step-number">3</div>
                    <div class="step-copy"><strong>Review the result</strong>Use the score as a prompt to investigate—not as a verdict.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        if len(article.strip()) < 30:
            st.warning("Please enter at least 30 characters of news text.")
        else:
            try:
                result = predict_news(get_model(), article)
                save_prediction(
                    article.strip(),
                    result["prediction"],
                    result["fake_probability"],
                    result["word_count"],
                )

                st.markdown("---")
                st.markdown('<div class="surface-label">02 / Analysis result</div>', unsafe_allow_html=True)
                result_class = "fake" if result["prediction"] == "Fake" else "real"
                st.markdown(
                    f"""
                    <div class="result-panel {result_class}">
                        <div class="result-kicker">Model prediction</div>
                        <div class="result-title">Likely {result["prediction"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                probability_column, words_column, characters_column = st.columns(3)
                probability_column.metric(
                    "Fake-class probability", f"{result['fake_probability']:.1%}"
                )
                words_column.metric("Words analyzed", f"{result['word_count']:,}")
                characters_column.metric("Characters", f"{len(article.strip()):,}")
                st.progress(result["fake_probability"])
                st.markdown(
                    '<p class="fine-print">The probability reflects the model\'s learned patterns and is not a verified probability that the article is false. Check important claims against primary sources and reputable independent reporting.</p>',
                    unsafe_allow_html=True,
                )
            except FileNotFoundError as error:
                st.error(str(error))
                st.code("python train_model.py")
            except ValueError as error:
                st.warning(str(error))

elif page == "Dashboard":
    render_page_intro(
        "Model insights • Evaluation",
        "Performance at a glance.",
        "Review how the production model performed on held-out examples from its dataset. These scores describe that evaluation—not guaranteed real-world reliability.",
    )
    metrics = load_metrics()
    comparison = load_comparison()
    if metrics is None or comparison.empty:
        st.warning("Model results are missing. Run `python train_model.py` first.")
    else:
        production = metrics["production_metrics"]
        score_columns = st.columns(4)
        score_columns[0].metric("Accuracy", f"{production['accuracy']:.2%}")
        score_columns[1].metric("Precision", f"{production['precision']:.2%}")
        score_columns[2].metric("Recall", f"{production['recall']:.2%}")
        score_columns[3].metric("F1-score", f"{production['f1_score']:.2%}")
        st.caption(
            f"Evaluation on {metrics['test_rows']:,} held-out articles. "
            "Results may vary on new sources, topics, and writing styles."
        )
        st.markdown("### Model comparison")
        chart_column, table_column = st.columns([1.3, 1], gap="large")
        with chart_column:
            chart_data = comparison.set_index("model")[
                ["accuracy", "precision", "recall", "f1_score"]
            ]
            st.bar_chart(chart_data, color=["#4ed9c1", "#79aaff", "#efba70", "#c18aff"])
        with table_column:
            st.dataframe(
                comparison.style.format(
                    {
                        "accuracy": "{:.2%}",
                        "precision": "{:.2%}",
                        "recall": "{:.2%}",
                        "f1_score": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Dataset split")
        dataset_columns = st.columns(3)
        dataset_columns[0].metric("All articles", f"{metrics['dataset_rows']:,}")
        dataset_columns[1].metric("Training set", f"{metrics['train_rows']:,}")
        dataset_columns[2].metric("Test set", f"{metrics['test_rows']:,}")
        with st.expander("How to read these metrics"):
            st.write("**Precision:** Of articles predicted fake, how many were labeled fake in the test data.")
            st.write("**Recall:** Of articles labeled fake in the test data, how many the model detected.")
            st.write("**F1-score:** The harmonic mean of precision and recall.")

elif page == "History":
    render_page_intro(
        "Your workspace • Saved analyses",
        "A record of your reviews.",
        "Search previously analyzed articles, inspect individual entries, or export your history for further review.",
    )
    history = get_history(limit=500)
    if history.empty:
        st.info("No prediction history yet. Analyze an article to start building your review log.")
    else:
        search_text = st.text_input("Search saved article text", placeholder="Type a word or phrase...")
        if search_text.strip():
            history = history[
                history["article"].str.contains(
                    search_text, case=False, regex=False, na=False
                )
            ]

        if history.empty:
            st.info("No matching articles found.")
        else:
            st.dataframe(
                history.drop(columns=["article"]),
                use_container_width=True,
                hide_index=True,
            )
            selected_id = st.selectbox(
                "Inspect an article",
                options=history["id"].tolist(),
                format_func=lambda record_id: f"Analysis #{record_id}",
            )
            selected = history[history["id"] == selected_id].iloc[0]
            safe_article = html.escape(selected["article"])
            safe_prediction = html.escape(selected["prediction"])
            st.markdown("### Selected article")
            st.markdown(
                f'<div class="surface"><div class="surface-label">Prediction: {safe_prediction} · Fake-class probability: {selected["fake_probability"]:.1%}</div><p>{safe_article}</p></div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Export visible history as CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="truthlens_history.csv",
                mime="text/csv",
            )
            st.markdown("---")
            st.warning("Clearing history permanently removes all saved records from the local database.")
            if st.button("Clear all history"):
                clear_history()
                st.rerun()

elif page == "About":
    render_page_intro(
        "About the project",
        "A lens on language.",
        "TruthLens is an educational news-text classification project built to make machine-learning outputs approachable and transparent.",
    )
    method_column, limits_column = st.columns(2, gap="large")
    with method_column:
        st.markdown(
            """
            <div class="surface">
                <div class="surface-label">The method</div>
                <h3>From words to a prediction</h3>
                <p>The app combines TF-IDF text features with a Logistic Regression classifier trained on labeled examples.</p>
                <div class="step-row"><div class="step-number">01</div><div class="step-copy"><strong>Input</strong>A headline or article is submitted for analysis.</div></div>
                <div class="step-row"><div class="step-number">02</div><div class="step-copy"><strong>Features</strong>The fitted vectorizer maps words and phrases into numeric features.</div></div>
                <div class="step-row"><div class="step-number">03</div><div class="step-copy"><strong>Output</strong>The classifier estimates a label and class probability.</div></div>
                <div class="step-row"><div class="step-number">04</div><div class="step-copy"><strong>History</strong>The prediction and submitted text are saved locally in SQLite.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with limits_column:
        st.markdown(
            """
            <div class="surface">
                <div class="surface-label">Important limitations</div>
                <h3>A prediction is not a fact-check</h3>
                <p>The model may learn patterns that do not correspond to factual accuracy. Treat its output as a starting point for independent research.</p>
                <div class="step-row"><div class="step-number">!</div><div class="step-copy"><strong>Dataset labels</strong>Training labels can be incomplete, biased, or incorrect.</div></div>
                <div class="step-row"><div class="step-number">!</div><div class="step-copy"><strong>Changing events</strong>New topics, sources, or writing styles may be misclassified.</div></div>
                <div class="step-row"><div class="step-number">!</div><div class="step-copy"><strong>Probability meaning</strong>Scores are model estimates, not the chance a claim is true.</div></div>
                <div class="step-row"><div class="step-number">!</div><div class="step-copy"><strong>Privacy</strong>Submitted text is stored in local prediction history.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Built with")
    st.write("Python · Streamlit · Pandas · Scikit-learn · TF-IDF · Logistic Regression · SQLite")
