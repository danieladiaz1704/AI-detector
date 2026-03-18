import { useState } from "react";

function App() {
  // State for input text
  const [text, setText] = useState("");

  // State for backend result
  const [result, setResult] = useState(null);

  // Loading state
  const [loading, setLoading] = useState(false);

  // Error state
  const [error, setError] = useState("");

  // Function to analyze text
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
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error("Failed to analyze text.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Backend connection error:", err);
      setError("Could not connect to the backend.");
    }

    setLoading(false);
  };

  return (
    <div style={styles.page}>
      <div style={styles.backgroundGlowOne}></div>
      <div style={styles.backgroundGlowTwo}></div>

      <div style={styles.container}>
        <div style={styles.headerCard}>
          <div style={styles.badge}>SMART AI ANALYZER</div>
          <h1 style={styles.title}>AI Text Detector</h1>
          <p style={styles.subtitle}>
            Paste your text and instantly see the probability of AI-generated
            versus human-written content.
          </p>
        </div>

        <div style={styles.mainGrid}>
          <div style={styles.inputCard}>
            <h2 style={styles.sectionTitle}>Input Text</h2>
            <p style={styles.sectionDescription}>
              Paste any paragraph, response, or written content below.
            </p>

            <textarea
              style={styles.textarea}
              rows="12"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your text here..."
            />

            {error && <p style={styles.errorText}>{error}</p>}

            <button
              style={loading ? styles.buttonDisabled : styles.button}
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Text"}
            </button>
          </div>

          <div style={styles.outputCard}>
            <h2 style={styles.sectionTitle}>Analysis Result</h2>
            <p style={styles.sectionDescription}>
              The system shows how likely the text is AI-generated or human-written.
            </p>

            {!result && !loading && (
              <div style={styles.placeholderBox}>
                <p style={styles.placeholderText}>
                  No analysis yet. Submit a text to view the result.
                </p>
              </div>
            )}

            {loading && (
              <div style={styles.placeholderBox}>
                <div style={styles.loader}></div>
                <p style={styles.placeholderText}>Processing your text...</p>
              </div>
            )}

            {result && (
              <div style={styles.resultBox}>
                <div style={styles.scoreCardAI}>
                  <p style={styles.scoreLabel}>AI Probability</p>
                  <h3 style={styles.scoreValue}>{result.ai_percentage}%</h3>
                  <div style={styles.barBackground}>
                    <div
                      style={{
                        ...styles.barFillAI,
                        width: `${result.ai_percentage}%`
                      }}
                    ></div>
                  </div>
                </div>

                <div style={styles.scoreCardHuman}>
                  <p style={styles.scoreLabel}>Human Probability</p>
                  <h3 style={styles.scoreValue}>{result.human_percentage}%</h3>
                  <div style={styles.barBackground}>
                    <div
                      style={{
                        ...styles.barFillHuman,
                        width: `${result.human_percentage}%`
                      }}
                    ></div>
                  </div>
                </div>

                <div style={styles.finalMessageBox}>
                  <p style={styles.finalMessage}>
                    {result.ai_percentage > result.human_percentage
                      ? "This text is more likely to be AI-generated."
                      : "This text is more likely to be human-written."}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(circle at top left, #0f172a 0%, #050816 45%, #02030a 100%)",
    position: "relative",
    overflow: "hidden",
    fontFamily: "Arial, sans-serif",
    padding: "40px 20px",
    color: "#ffffff"
  },
  backgroundGlowOne: {
    position: "absolute",
    top: "-120px",
    left: "-120px",
    width: "380px",
    height: "380px",
    background: "rgba(59, 130, 246, 0.18)",
    filter: "blur(120px)",
    borderRadius: "50%"
  },
  backgroundGlowTwo: {
    position: "absolute",
    bottom: "-120px",
    right: "-120px",
    width: "380px",
    height: "380px",
    background: "rgba(168, 85, 247, 0.18)",
    filter: "blur(120px)",
    borderRadius: "50%"
  },
  container: {
    maxWidth: "1200px",
    margin: "0 auto",
    position: "relative",
    zIndex: 2
  },
  headerCard: {
    textAlign: "center",
    marginBottom: "30px",
    padding: "30px 20px"
  },
  badge: {
    display: "inline-block",
    padding: "8px 14px",
    borderRadius: "999px",
    border: "1px solid rgba(96, 165, 250, 0.35)",
    background: "rgba(96, 165, 250, 0.08)",
    color: "#7dd3fc",
    fontSize: "12px",
    fontWeight: "bold",
    letterSpacing: "1.6px",
    marginBottom: "18px"
  },
  title: {
    fontSize: "64px",
    margin: "0 0 12px 0",
    fontWeight: "bold",
    lineHeight: 1.05,
    background: "linear-gradient(90deg, #ffffff, #93c5fd, #c084fc)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent"
  },
  subtitle: {
    maxWidth: "760px",
    margin: "0 auto",
    color: "#cbd5e1",
    fontSize: "18px",
    lineHeight: "1.7"
  },
  mainGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "24px"
  },
  inputCard: {
    background: "rgba(15, 23, 42, 0.72)",
    border: "1px solid rgba(148, 163, 184, 0.16)",
    borderRadius: "22px",
    padding: "28px",
    backdropFilter: "blur(14px)",
    boxShadow: "0 10px 35px rgba(0, 0, 0, 0.30)"
  },
  outputCard: {
    background: "rgba(15, 23, 42, 0.72)",
    border: "1px solid rgba(148, 163, 184, 0.16)",
    borderRadius: "22px",
    padding: "28px",
    backdropFilter: "blur(14px)",
    boxShadow: "0 10px 35px rgba(0, 0, 0, 0.30)"
  },
  sectionTitle: {
    fontSize: "26px",
    marginBottom: "8px",
    color: "#f8fafc"
  },
  sectionDescription: {
    fontSize: "15px",
    color: "#94a3b8",
    marginBottom: "18px",
    lineHeight: "1.6"
  },
  textarea: {
    width: "100%",
    minHeight: "280px",
    padding: "18px",
    borderRadius: "16px",
    border: "1px solid rgba(148, 163, 184, 0.25)",
    background: "rgba(2, 6, 23, 0.65)",
    color: "#f8fafc",
    fontSize: "16px",
    lineHeight: "1.6",
    resize: "none",
    outline: "none",
    boxSizing: "border-box"
  },
  button: {
    marginTop: "20px",
    width: "100%",
    padding: "15px 18px",
    borderRadius: "14px",
    border: "none",
    background: "linear-gradient(90deg, #2563eb, #7c3aed)",
    color: "#ffffff",
    fontSize: "16px",
    fontWeight: "bold",
    cursor: "pointer",
    boxShadow: "0 0 25px rgba(99, 102, 241, 0.35)"
  },
  buttonDisabled: {
    marginTop: "20px",
    width: "100%",
    padding: "15px 18px",
    borderRadius: "14px",
    border: "none",
    background: "linear-gradient(90deg, #334155, #475569)",
    color: "#cbd5e1",
    fontSize: "16px",
    fontWeight: "bold",
    cursor: "not-allowed"
  },
  errorText: {
    marginTop: "12px",
    color: "#fca5a5",
    fontSize: "14px"
  },
  placeholderBox: {
    minHeight: "320px",
    borderRadius: "18px",
    border: "1px dashed rgba(148, 163, 184, 0.28)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    color: "#94a3b8",
    padding: "20px",
    textAlign: "center"
  },
  placeholderText: {
    fontSize: "15px",
    lineHeight: "1.6"
  },
  loader: {
    width: "46px",
    height: "46px",
    border: "4px solid rgba(255,255,255,0.15)",
    borderTop: "4px solid #60a5fa",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
    marginBottom: "16px"
  },
  resultBox: {
    display: "flex",
    flexDirection: "column",
    gap: "20px"
  },
  scoreCardAI: {
    background: "rgba(2, 6, 23, 0.55)",
    border: "1px solid rgba(255, 80, 80, 0.35)",
    borderRadius: "18px",
    padding: "24px",
    boxShadow: "0 0 25px rgba(255, 80, 80, 0.12)"
  },
  scoreCardHuman: {
    background: "rgba(2, 6, 23, 0.55)",
    border: "1px solid rgba(72, 255, 163, 0.35)",
    borderRadius: "18px",
    padding: "24px",
    boxShadow: "0 0 25px rgba(72, 255, 163, 0.12)"
  },
  scoreLabel: {
    fontSize: "14px",
    color: "#cbd5e1",
    marginBottom: "10px",
    textTransform: "uppercase",
    letterSpacing: "1px"
  },
  scoreValue: {
    fontSize: "42px",
    margin: "0 0 18px 0",
    color: "#ffffff"
  },
  barBackground: {
    width: "100%",
    height: "14px",
    borderRadius: "999px",
    background: "rgba(148, 163, 184, 0.18)",
    overflow: "hidden"
  },
  barFillAI: {
    height: "100%",
    borderRadius: "999px",
    background: "linear-gradient(90deg, #ef4444, #fb7185)"
  },
  barFillHuman: {
    height: "100%",
    borderRadius: "999px",
    background: "linear-gradient(90deg, #22c55e, #4ade80)"
  },
  finalMessageBox: {
    background: "rgba(15, 23, 42, 0.72)",
    border: "1px solid rgba(148, 163, 184, 0.16)",
    borderRadius: "18px",
    padding: "20px",
    textAlign: "center"
  },
  finalMessage: {
    margin: 0,
    fontSize: "18px",
    color: "#f8fafc",
    fontWeight: "bold"
  }
};

export default App;