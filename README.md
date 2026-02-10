# A-Share Return Prediction & Market-Neutral Portfolio Optimization

This repository contains the implementation of my Bachelor's Thesis: **"Research on Financial Data Correlation Based on Deep Learning"**. It features a hybrid Transformer model for stock return prediction and an enhanced Mean-Variance framework for portfolio optimization in the Chinese A-share market.

## 🚀 Key Highlights
- **Model Architecture:** Transformer with **Convolutional Positional Encoding (CPE)** to better capture local temporal dependencies in high-frequency financial time series.
- **Risk Management:** Implementation of a **Market-Neutral** strategy, effectively decoupling portfolio performance from broad market volatility.
- **Performance:** Achieved a **Sharpe Ratio of 1.99** and a **Beta of 0.06** in historical backtesting.

---

## 📊 Methodology

### 1. Data Pipeline
- **Universe:** A-share market (Main board, CSI 300 components).
- **Preprocessing:** Z-score normalization, outlier removal (winsorization), and handling of missing values.
- **Features:** Technical indicators (Moving Averages, RSI, MACD) combined with lagged return data.

### 2. Model: Conv-Transformer
Traditional Transformers use absolute/fixed positional encoding. I introduced **Convolutional Positional Encoding** to:
- Capture local spatial-temporal correlations within local windows.
- Improve the model's sensitivity to short-term price fluctuations in the A-share market.

### 3. Portfolio Optimization
Improved the **Classical Mean-Variance Model** by:
- Integrating DL-predicted returns as expected return vectors ($E[R]$).
- Applying weight constraints to ensure a Market-Neutral profile (Beta minimization).

---

## 📈 Backtesting Results
The strategy was backtested using a rolling-window approach.

| Metric | Value |
| :--- | :--- |
| **Annualized Sharpe Ratio** | **1.99** |
| **Beta (Market Sensitivity)** | **0.06** |
| **Max Drawdown (Monthly Avg)** | **10%** |
| **Directional Accuracy** | **91.64%** |

---

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, TensorFlow, NumPy, Pandas, 
- **Optimization:** Gurobi (for Mean-Variance Quadratic Programming)
- **Environment:** Python, CUDA
## 📁 Repository Structure
```text
├── data/               # Sample data / Data loaders
├── models/             # Transformer & CPE implementation
├── optimization/       # Mean-Variance & Portfolio scripts
├── backtest/           # Performance evaluation & metrics
├── main.py             # Entry point for training and testing
└── requirements.txt    # Dependencies
