# 🔭 Technical Convergence Terminal

A high-fidelity financial analytics dashboard that utilizes multi-indicator convergence and cross-timeframe analysis to identify high-probability market entries. Built with **Python**, **Streamlit**, and **Plotly**.



## 🧠 The Concept: Why "Convergence"?
In technical analysis, a single indicator often provides "false positives." This terminal solves that by requiring **Convergence**—the agreement of three distinct mathematical models—before issuing a high-confidence signal:
1.  **Momentum:** MACD Crossover (Moving Average Convergence Divergence).
2.  **Relative Strength:** Wilder’s RSI filtering (ensures price isn't overextended).
3.  **Trend Intensity:** ADX (Average Directional Index) to confirm trend strength > 25.

---

## 🚀 Key Features

### **1. Multi-Tiered Signal Engine**
The terminal categorizes price action into three distinct logic tiers to help traders distinguish between "noise" and "intent":
* **Strong (Diamond):** Full convergence of MACD, RSI, and ADX.
* **Standard (Triangle):** MACD crossover validated by RSI.
* **Pure (Circle):** Raw MACD momentum for early trend detection.

### **2. Cross-Timeframe Anchor**
The system automatically fetches and analyzes **Daily** data in the background. It identifies a "Daily Anchor Trend" to warn users if their lower-interval trade (e.g., 15m or 1h) is fighting the primary macro direction.

### **3. Professional Visualization**
* **Dynamic Headers:** The terminal automatically updates titles based on the active ticker input.
* **Wilder’s RSI Implementation:** Uses the historically accurate Wilder’s smoothing method for RSI calculations.
* **Unified Hover:** Synchronized crosshairs across all sub-charts for precise data correlation.

---

## 🛠️ Installation & Setup

### **1. Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/technical-convergence-terminal.git](https://github.com/YOUR_USERNAME/technical-convergence-terminal.git)
cd technical-convergence-terminal
