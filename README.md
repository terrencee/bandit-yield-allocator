# Bandit Allocation in Indian Yield Curve

📍 **Live Demo:** [Streamlit App](https://bandit-yield-allocator-bandit-adhiraj.streamlit.app/)  
*(replace with your actual Streamlit Cloud link after deployment)*

This project explores **multi-armed bandit algorithms** (ε-Greedy and LinUCB) for daily allocation between  
- **91-day Treasury Bills**,  
- **364-day Treasury Bills**, and  
- **10-year Government Securities (G-Secs)**.  

The app is built in **Streamlit** and provides interactive parameter tuning, baseline comparisons, and visualizations.

---

## ⚙️ Features
- Load pre-cleaned daily yield data (`data/yields_daily_91d_364d_10y.csv`).
- Define reward proxies with duration penalties and switching costs.
- Compare baselines vs. ε-Greedy vs. LinUCB bandits.
- Interactive plots:
  - Equity curves (terminal wealth, Sharpe, return/vol).
  - Allocation timeline (compressed runs).
  - Rolling allocation share.
- Adjustable date range window.

---

## 📦 Requirements
See [`requirements.txt`](requirements.txt).  
Main dependencies:
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `plotly`

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
