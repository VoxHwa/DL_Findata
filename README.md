# A-Share Return Prediction & Portfolio Optimization

This repository contains the implementation of my Bachelor's Thesis: **"Research on Financial Data Correlation Based on Deep Learning"**. It features a hybrid Transformer model for stock return prediction and an enhanced Mean-Variance framework for portfolio optimization in the Chinese A-share market.

## 🚀 Key Highlights
- **Model Architecture:** Transformer with **Convolutional Positional Encoding (CPE)** to better capture local temporal dependencies in high-frequency financial time series. The best model predicts with a **direction accuracy of 91.64% and MAE of 0.588(1e-2)**.
- **Improved Mean-Variance model in A-share market:** Achieved a **Sharpe Ratio of 1.99** and a **Beta of 0.06** in historical backtesting (with alpha=1e-3).

---

## 📊 Methodology

### 1. Data Pipeline
- Data is collected from BigQuant using SQL commands.
- **Universe:** A-share market.
- **Preprocessing:** Z-score normalization (极差标准化 doesn't work well during the test phrase), outlier removal, and handling of missing values.
- **Features:** <img width="1178" height="886" alt="image" src="https://github.com/user-attachments/assets/9a0a3b9f-3424-4769-8221-b392de24ff25" />


### 2. Model: Conv-Transformer
Traditional Transformers use absolute/fixed positional encoding. I introduced **Convolutional Positional Encoding** to:
- Capture local spatial-temporal correlations within local windows.
- Improve the model's sensitivity to short-term price fluctuations in the A-share market.
- <img width="964" height="369" alt="image" src="https://github.com/user-attachments/assets/625c366d-7bcc-4f8c-9e85-4aee96731a38" />


### 3. Portfolio Optimization
<img width="459" height="238" alt="image" src="https://github.com/user-attachments/assets/8b6f5c5f-9d94-48dc-b378-3c11ee993370" />

<img width="1000" height="609" alt="image" src="https://github.com/user-attachments/assets/3587755a-b8bc-4be7-8076-8a8a41171ef1" />

<img width="966" height="373" alt="image" src="https://github.com/user-attachments/assets/34a425db-3254-4517-a27b-218cadd615f2" />

<img width="840" height="398" alt="image" src="https://github.com/user-attachments/assets/ad3050aa-18c2-457b-8479-a0953265aa8f" />

---

## 📈 Backtesting Results
<img width="2843" height="1442" alt="Profit" src="https://github.com/user-attachments/assets/bd443d5a-998e-44e8-8ef0-51edc388b40d" />

<img width="878" height="292" alt="image-20260210173023767" src="https://github.com/user-attachments/assets/26cbc874-a360-4ff8-9ec9-5b7b74ea3274" />

<img width="4400" height="2475" alt="max_drawdown" src="https://github.com/user-attachments/assets/9f777134-1b4c-439d-8734-6a6739ae238d" />

---

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, TensorFlow, NumPy, Pandas, Scipy, keras, sklearn
- **Optimization:** Gurobi (for Mean-Variance Quadratic Programming)
- **Environment:** Python, CUDA
