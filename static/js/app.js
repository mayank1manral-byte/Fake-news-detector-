const pages = ["analyze", "dashboard", "history", "about"];
const pageTitles = {
  analyze: "ANALYZE",
  dashboard: "DASHBOARD",
  history: "HISTORY",
  about: "ABOUT",
};

const byId = (id) => document.getElementById(id);

function showPage(name) {
  for (const page of pages) {
    byId(`page-${page}`).classList.toggle("active", page === name);
    document.querySelectorAll(`[data-page-link="${page}"]`).forEach((link) => {
      if (link.classList.contains("nav-link")) {
        link.classList.toggle("active", page === name);
        link.setAttribute("aria-current", page === name ? "page" : "false");
      }
    });
  }
  byId("current-section").textContent = pageTitles[name];
  if (name === "dashboard") loadMetrics();
  if (name === "history") loadHistory();
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("The server returned an unreadable response. Please try again.");
  }
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}

function setFormMessage(message) {
  const element = byId("form-message");
  element.textContent = message;
  element.classList.toggle("hidden", !message);
}

function updateInputCount() {
  const text = byId("article-input").value;
  const length = text.trim().length;
  byId("character-count").textContent = text.length.toLocaleString();
  const feedback = byId("input-feedback");
  const meta = feedback.parentElement;
  const isShort = length > 0 && length < 30;
  meta.classList.toggle("invalid", isShort);
  feedback.textContent = isShort ? `${30 - length} more characters needed` : "At least 30 characters";
}

byId("article-input").addEventListener("input", updateInputCount);
byId("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const article = byId("article-input").value.trim();
  const button = byId("analyze-button");
  byId("result-card").classList.add("hidden");
  setFormMessage("");
  if (article.length < 30) {
    setFormMessage("Please enter at least 30 characters of news text.");
    byId("article-input").focus();
    return;
  }
  if (article.length > 50000) {
    setFormMessage("Please keep the news text below 50,000 characters.");
    return;
  }

  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Analyzing...";
  try {
    const result = await requestJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article }),
    });
    const isFake = result.prediction === "Fake";
    const card = byId("result-card");
    card.classList.toggle("fake", isFake);
    byId("result-verdict").textContent = `Likely ${result.prediction}`;
    byId("fake-probability").textContent = `${(result.fake_probability * 100).toFixed(1)}%`;
    byId("word-count").textContent = result.word_count.toLocaleString();
    byId("result-character-count").textContent = result.character_count.toLocaleString();
    byId("probability-fill").value = Math.max(0, Math.min(1, result.fake_probability));
    card.classList.remove("hidden");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    setFormMessage(error.message);
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = "Analyze article";
  }
});

function addCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (className) cell.className = className;
  row.appendChild(cell);
  return cell;
}

async function loadMetrics() {
  const grid = byId("metric-grid");
  const body = byId("comparison-body");
  const empty = byId("metrics-empty");
  try {
    const { metrics, comparison } = await requestJson("/api/metrics");
    grid.replaceChildren();
    body.replaceChildren();
    if (!metrics || !comparison.length) {
      empty.classList.remove("hidden");
      return;
    }
    empty.classList.add("hidden");
    const values = metrics.production_metrics;
    for (const [label, key] of [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1-score", "f1_score"]]) {
      const card = document.createElement("div");
      card.className = "metric-card";
      const name = document.createElement("span");
      name.textContent = label;
      const value = document.createElement("strong");
      value.textContent = `${(values[key] * 100).toFixed(2)}%`;
      card.append(name, value);
      grid.appendChild(card);
    }
    byId("metrics-caption").textContent = `Evaluation on ${metrics.test_rows.toLocaleString()} held-out articles. Results may vary on new sources, topics, and writing styles.`;
    for (const item of comparison) {
      const row = document.createElement("tr");
      addCell(row, item.model);
      for (const key of ["accuracy", "precision", "recall", "f1_score"]) {
        addCell(row, `${(item[key] * 100).toFixed(2)}%`);
      }
      body.appendChild(row);
    }
  } catch (error) {
    empty.textContent = error.message;
    empty.classList.remove("hidden");
  }
}

let historyTimeout;
byId("history-search").addEventListener("input", () => {
  clearTimeout(historyTimeout);
  historyTimeout = setTimeout(loadHistory, 180);
});

async function loadHistory() {
  const query = byId("history-search").value;
  const body = byId("history-body");
  const empty = byId("history-empty");
  const selected = byId("selected-article");
  byId("export-history").href = `/api/history.csv?q=${encodeURIComponent(query)}`;
  body.replaceChildren();
  selected.classList.add("hidden");
  try {
    const data = await requestJson(`/api/history?q=${encodeURIComponent(query)}`);
    empty.classList.toggle("hidden", data.records.length > 0);
    if (!data.records.length) {
      empty.textContent = query ? "No matching articles found." : "No prediction history yet. Analyze an article to start your review log.";
      return;
    }
    for (const record of data.records) {
      const row = document.createElement("tr");
      addCell(row, `#${record.id}`);
      addCell(row, record.created_at);
      const prediction = addCell(row, "");
      const pill = document.createElement("span");
      pill.className = `pill${record.prediction === "Fake" ? " fake" : ""}`;
      pill.textContent = record.prediction;
      prediction.appendChild(pill);
      addCell(row, `${(record.fake_probability * 100).toFixed(1)}%`);
      addCell(row, Number(record.word_count).toLocaleString());
      const action = addCell(row, "");
      const inspect = document.createElement("button");
      inspect.className = "table-action";
      inspect.type = "button";
      inspect.textContent = "Inspect";
      inspect.addEventListener("click", () => showArticle(record));
      action.appendChild(inspect);
      body.appendChild(row);
    }
  } catch (error) {
    empty.textContent = error.message;
    empty.classList.remove("hidden");
  }
}

function showArticle(record) {
  byId("selected-meta").textContent = `Prediction: ${record.prediction} · Fake-class probability: ${(record.fake_probability * 100).toFixed(1)}%`;
  byId("article-preview").textContent = record.article;
  byId("selected-article").classList.remove("hidden");
  byId("selected-article").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

byId("clear-history").addEventListener("click", async () => {
  if (!window.confirm("Permanently clear all saved prediction history?")) return;
  try {
    await requestJson("/api/history", { method: "DELETE" });
    await loadHistory();
  } catch (error) {
    window.alert(error.message);
  }
});

document.querySelectorAll("[data-page-link]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    showPage(link.dataset.pageLink);
  });
});
