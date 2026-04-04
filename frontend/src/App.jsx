import { useState, useRef, useEffect } from "react";

const MAX_CHARS   = 5000;
const HISTORY_KEY = "ai_detector_history";
const MAX_HISTORY = 10;

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entries) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
}

function App() {
  const [text,          setText]          = useState("");
  const [selectedModel, setSelectedModel] = useState("lstm");
  const [result,        setResult]        = useState(null);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState("");
  const [history,       setHistory]       = useState(loadHistory);
  const [showHistory,   setShowHistory]   = useState(false);
  const [copied,        setCopied]        = useState(false);

  const fileInputRef = useRef(null);

  const wordCount = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
  const charCount = text.length;

  // ── File upload ─────────────────────────────────────────
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith(".txt")) {
      setError("Only .txt files are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = ev.target.result.slice(0, MAX_CHARS);
      setText(content);
      setError("");
      setResult(null);
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  // ── Analysis ─────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter some text before analyzing.");
      setResult(null);
      return;
    }
    if (charCount > MAX_CHARS) {
      setError(`Text is too long. Maximum is ${MAX_CHARS.toLocaleString()} characters.`);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_name: selectedModel }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to analyze text.");
      }

      setResult(data);

      // Persist to history
      const entry = {
        id            : Date.now(),
        timestamp     : new Date().toLocaleString(),
        model         : data.model_used,
        ai_probability: data.ai_probability,
        human_probability: data.human_probability,
        preview       : text.slice(0, 80).trim() + (text.length > 80 ? "…" : ""),
        full_result   : data,
      };
      const updated = [entry, ...history];
      setHistory(updated);
      saveHistory(updated);

    } catch (err) {
      setError(err.message || "Something went wrong. Is the backend running?");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // ── Export ───────────────────────────────────────────────
  const handleExport = () => {
    if (!result) return;
    const payload = JSON.stringify(
      { text_analyzed: text, ...result },
      null,
      2
    );
    navigator.clipboard.writeText(payload).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleDownload = () => {
    if (!result) return;
    const payload = JSON.stringify({ text_analyzed: text, ...result }, null, 2);
    const blob    = new Blob([payload], { type: "application/json" });
    const url     = URL.createObjectURL(blob);
    const a       = document.createElement("a");
    a.href        = url;
    a.download    = `ai_detection_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Sentence highlighting ────────────────────────────────
  const renderHighlighting = () => {
    if (!result?.sentences?.length) {
      return (
        <p className="placeholder-text">
          No highlighted text yet. Analyze your text to see sentence-level results.
        </p>
      );
    }
    return result.sentences.map((part, i) => (
      <span
        key={i}
        className={`highlight-span ${part.label === "AI" ? "highlight-ai" : "highlight-human"}`}
        title={`${part.label} — ${(part.probability * 100).toFixed(1)}%`}
      >
        {part.sentence}{" "}
      </span>
    ));
  };

  const aiPct    = result?.ai_probability    ?? 0;
  const humanPct = result?.human_probability ?? 0;

  return (
    <div className="page">
      <div className="background-glow-one" />
      <div className="background-glow-two" />

      <div className="container">

        {/* ── Header ─────────────────────────────────────── */}
        <div className="header-card">
          <div className="badge">SMART AI ANALYZER</div>
          <h1 className="title">AI Text Detector</h1>
          <p className="subtitle">
            Paste text or upload a .txt file and instantly detect AI-generated
            content across four deep learning models.
          </p>
        </div>

        {/* ── Main grid ──────────────────────────────────── */}
        <div className="main-grid">

          {/* Left — Input */}
          <div className="card">
            <h2 className="section-title">Input Text</h2>
            <p className="section-description">
              Paste any paragraph, essay, or article below, or upload a .txt file.
            </p>

            {/* Model select */}
            <label className="field-label">Choose Model</label>
            <select
              className="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="traditional">Logistic Regression</option>
              <option value="lstm">LSTM</option>
              <option value="gru">GRU</option>
              <option value="bert">BERT</option>
              <option value="ensemble">Ensemble (all models)</option>
            </select>

            {/* Text label + upload */}
            <div className="label-row">
              <label className="field-label" style={{ margin: 0 }}>Text to Analyze</label>
              <button
                className="upload-btn"
                onClick={() => fileInputRef.current?.click()}
                title="Upload a .txt file"
              >
                Upload .txt
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
            </div>

            <textarea
              className="textarea"
              rows="10"
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
              placeholder="Paste your text here or upload a .txt file…"
            />

            {/* Counter */}
            <div className="counter-row">
              <span className={charCount > MAX_CHARS * 0.9 ? "counter counter-warn" : "counter"}>
                {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()} chars
              </span>
              <span className="counter">{wordCount.toLocaleString()} words</span>
            </div>

            {error && <p className="error-text">{error}</p>}

            <button
              className={loading ? "button button-disabled" : "button"}
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? "Analyzing…" : "Analyze Text"}
            </button>
          </div>

          {/* Right — Results */}
          <div className="card">
            <h2 className="section-title">Analysis Result</h2>
            <p className="section-description">
              AI vs human probability, sentence-level highlighting, and export.
            </p>

            {!result && !loading && (
              <div className="placeholder-box">
                <p className="placeholder-text">
                  No analysis yet. Submit text to view results.
                </p>
              </div>
            )}

            {loading && (
              <div className="placeholder-box">
                <div className="loader" />
                <p className="placeholder-text">Processing your text…</p>
              </div>
            )}

            {result && (
              <div className="result-box">

                {/* Scores */}
                <div className="score-card-ai">
                  <p className="score-label">AI Probability</p>
                  <h3 className="score-value">{aiPct}%</h3>
                  <div className="bar-background">
                    <div className="bar-fill-ai" style={{ width: `${aiPct}%` }} />
                  </div>
                </div>

                <div className="score-card-human">
                  <p className="score-label">Human Probability</p>
                  <h3 className="score-value">{humanPct}%</h3>
                  <div className="bar-background">
                    <div className="bar-fill-human" style={{ width: `${humanPct}%` }} />
                  </div>
                </div>

                {/* Verdict */}
                <div className="final-message-box">
                  <p className="final-message">
                    {aiPct > humanPct
                      ? "This text is more likely to be AI-generated."
                      : "This text is more likely to be human-written."}
                  </p>
                  <p className="meta-text">
                    Model: {result.model_used?.toUpperCase()}
                    {result.processing_time_s && ` · ${result.processing_time_s}s`}
                  </p>
                </div>

                {/* Export buttons */}
                <div className="export-row">
                  <button className="export-btn" onClick={handleExport}>
                    {copied ? "✓ Copied!" : "Copy JSON"}
                  </button>
                  <button className="export-btn" onClick={handleDownload}>
                    Download JSON
                  </button>
                </div>

                {/* Sentence highlighting */}
                <div className="highlight-box">
                  <p className="highlight-title">Text Highlighting</p>
                  <div className="highlighted-text">{renderHighlighting()}</div>
                  <div className="legend">
                    <div className="legend-item">
                      <span className="legend-color legend-human" />
                      <span className="legend-text">Human-like</span>
                    </div>
                    <div className="legend-item">
                      <span className="legend-color legend-ai" />
                      <span className="legend-text">AI-like</span>
                    </div>
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>

        {/* ── History panel ───────────────────────────────── */}
        {history.length > 0 && (
          <div className="history-section">
            <button
              className="history-toggle"
              onClick={() => setShowHistory((v) => !v)}
            >
              {showHistory ? "▲ Hide" : "▼ Show"} Analysis History ({history.length})
            </button>

            {showHistory && (
              <div className="history-grid">
                {history.map((entry) => (
                  <div
                    key={entry.id}
                    className="history-card"
                    onClick={() => {
                      setResult(entry.full_result);
                      setError("");
                    }}
                    title="Click to reload this result"
                  >
                    <div className="history-header">
                      <span className="history-model">{entry.model?.toUpperCase()}</span>
                      <span className="history-time">{entry.timestamp}</span>
                    </div>
                    <p className="history-preview">{entry.preview}</p>
                    <div className="history-scores">
                      <span className="history-ai">AI {entry.ai_probability}%</span>
                      <span className="history-human">Human {entry.human_probability}%</span>
                    </div>
                  </div>
                ))}

                <button
                  className="clear-history-btn"
                  onClick={() => {
                    setHistory([]);
                    saveHistory([]);
                  }}
                >
                  Clear History
                </button>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;