import { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState("traditional");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter some text before analyzing.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text,
          model_name: selectedModel
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to analyze text.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const renderHighlighting = () => {
    if (!result || !result.sentences || result.sentences.length === 0) {
      return (
        <p className="placeholder-text">
          No highlighted text yet. Analyze your text to see sentence-level results.
        </p>
      );
    }

    return result.sentences.map((part, index) => {
      const isAI = part.label === "AI";

      return (
        <span
          key={index}
          className={`highlight-span ${isAI ? "highlight-ai" : "highlight-human"}`}
        >
          {part.sentence}{" "}
        </span>
      );
    });
  };

  const aiPercentage = result ? result.ai_probability : 0;
  const humanPercentage = result ? result.human_probability : 0;

  return (
    <div className="page">
      <div className="background-glow-one"></div>
      <div className="background-glow-two"></div>

      <div className="container">
        <div className="header-card">
          <div className="badge">SMART AI ANALYZER</div>
          <h1 className="title">AI Text Detector</h1>
          <p className="subtitle">
            Paste your text and instantly see the probability of AI-generated
            versus human-written content, along with highlighted text sections.
          </p>
        </div>

        <div className="main-grid">
          <div className="card">
            <h2 className="section-title">Input Text</h2>
            <p className="section-description">
              Paste any paragraph, response, or written content below.
            </p>

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
            </select>

            <label className="field-label">Text to Analyze</label>
            <textarea
              className="textarea"
              rows="10"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your text here..."
            />

            {error && <p className="error-text">{error}</p>}

            <button
              className={loading ? "button button-disabled" : "button"}
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Text"}
            </button>
          </div>

          <div className="card">
            <h2 className="section-title">Analysis Result</h2>
            <p className="section-description">
              The system shows the probability of AI-generated and human-written
              text, plus highlighted text sections.
            </p>

            {!result && !loading && (
              <div className="placeholder-box">
                <p className="placeholder-text">
                  No analysis yet. Submit a text to view the result.
                </p>
              </div>
            )}

            {loading && (
              <div className="placeholder-box">
                <div className="loader"></div>
                <p className="placeholder-text">Processing your text...</p>
              </div>
            )}

            {result && (
              <div className="result-box">
                <div className="score-card-ai">
                  <p className="score-label">AI Probability</p>
                  <h3 className="score-value">{aiPercentage}%</h3>
                  <div className="bar-background">
                    <div
                      className="bar-fill-ai"
                      style={{ width: `${aiPercentage}%` }}
                    ></div>
                  </div>
                </div>

                <div className="score-card-human">
                  <p className="score-label">Human Probability</p>
                  <h3 className="score-value">{humanPercentage}%</h3>
                  <div className="bar-background">
                    <div
                      className="bar-fill-human"
                      style={{ width: `${humanPercentage}%` }}
                    ></div>
                  </div>
                </div>

                <div className="final-message-box">
                  <p className="final-message">
                    {aiPercentage > humanPercentage
                      ? "This text is more likely to be AI-generated."
                      : "This text is more likely to be human-written."}
                  </p>
                </div>

                <div className="highlight-box">
                  <p className="highlight-title">Text Highlighting</p>

                  <div className="highlighted-text">{renderHighlighting()}</div>

                  <div className="legend">
                    <div className="legend-item">
                      <span className="legend-color legend-human"></span>
                      <span className="legend-text">More human-written</span>
                    </div>

                    <div className="legend-item">
                      <span className="legend-color legend-ai"></span>
                      <span className="legend-text">More AI-like</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;