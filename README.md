# GAT-TFT Model for Sector Correlation Analysis in Indian Equity Market

## 🎯 Project Overview

This project demonstrates a **hybrid deep learning architecture** combining **Graph Attention Networks (GAT)** and **Temporal Fusion Transformer (TFT)** to analyze sector-level correlations in the Indian stock market. The model predicts stock returns and price movements by examining cross-sectoral relationships and temporal dynamics.

### Key Innovation
The hybrid approach leverages:
- **Graph Attention Networks**: Captures inter-sector correlations and dependencies
- **Temporal Fusion Transformer**: Models temporal patterns and future predictions
- **Sector-Aware Analysis**: Understands sector-specific dynamics and their impact on individual stocks

---

## 📊 Datasets

### Data Source
- **Primary Source**: Yahoo Finance API (`yfinance`)
- **Secondary Source**: Kaggle NSE Datasets (optional for enhanced historical data)

### Stock Universe (250+ Stocks)
The project includes comprehensive coverage across Indian equity indices:

| Index | Coverage | Count |
|-------|----------|-------|
| **Nifty 50** | Large Cap | 50 |
| **Nifty Next 50** | Large Cap | 50 |
| **Nifty Midcap 100** | Mid Cap | ~100+ |
| **Sectoral Stocks** | Cross-sector | 50+ |

### Sector Coverage
The model covers **19 major sectors** including:
- Information Technology
- Banking & Financial Services
- Pharmaceuticals & Healthcare
- Automobiles & Auto Components
- FMCG & Consumer Services
- Energy & Power
- Metals & Mining
- Cement & Infrastructure
- Chemicals & Textiles
- Telecom & Capital Goods

### Data Specifications
- **Time Period**: 10+ years of historical data
- **Frequency**: Daily OHLCV (Open, High, Low, Close, Volume)
- **Data Points**: ~300,000+ individual observations
- **Features per Stock**: 5 base features + technical indicators

### Technical Indicators Used
The preprocessing includes advanced technical indicators:

1. **Simple Moving Average (SMA)**: Trend identification
2. **Exponential Moving Average (EMA)**: Responsive trend analysis
3. **Relative Strength Index (RSI)**: Momentum measurement
4. **MACD**: Trend momentum and divergence
5. **Bollinger Bands**: Volatility measurement
6. **Average True Range (ATR)**: Market volatility tracking

---

## 🧠 Deep Learning Framework & Architecture

### Core Libraries
```
PyTorch 2.x              - Deep Learning Framework
PyTorch Geometric        - Graph Neural Networks
PyTorch Sparse           - Sparse tensor operations
NumPy & Pandas           - Data manipulation
Scikit-learn             - ML utilities
```

### Model Architecture

#### 1. **Graph Attention Network (GAT) - Sector Layer**
```
Input: Sector correlation matrix (19 sectors)
├── Multi-Head Attention (8 heads)
│   ├── Captures sector interdependencies
│   ├── Learns weighted relationships between sectors
│   └── Produces attention weights for sector pairs
├── Graph Convolution
│   └── Aggregates sector-level information
└── Output: Sector-aware embeddings (batch_size × sectors × hidden_dim)
```

**Purpose**: Model how sectors influence each other through market correlations

#### 2. **Temporal Fusion Transformer (TFT) - Time Series Layer**
```
Input: Historical OHLCV + Technical Indicators (time_steps × features)
├── Variable Selection Network
│   ├── Learns importance of each feature
│   └── Produces feature weights
├── LSTM Encoder
│   ├── Captures temporal dependencies
│   └── Produces context vector
├── Multi-Head Attention
│   ├── Self-attention on historical data
│   ├── Cross-attention with sector context
│   └── Produces temporal attention patterns
├── Decoder
│   ├── Generates predictions
│   └── Produces prediction intervals
└── Output: Stock returns/movements (batch_size × forecast_horizon)
```

**Purpose**: Capture temporal patterns and adapt to sector dynamics

#### 3. **Hybrid Integration**
```
┌─────────────────────────────────────────┐
│    Sector Graph (GAT)                   │
│  (Correlation Analysis)                 │
└──────────────┬──────────────────────────┘
               │ Sector Context
               ▼
┌─────────────────────────────────────────┐
│ Temporal Fusion Transformer (TFT)       │
│ Stock-level Prediction with             │
│ Sector-Aware Attention                  │
└──────────────┬──────────────────────────┘
               │
               ▼
        Prediction Output
    (Returns & Movements)
```

### Model Variants Included

| File | Version | Purpose |
|------|---------|---------|
| `model_architecture.py` | v1 | Basic GAT-TFT architecture |
| `model_architecture_v2.py` | v2 | Enhanced attention mechanisms |
| `model_v3_sector_aware.py` | v3 | Sector-aware improvements |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd GAT_TFT-model-for-sector-correlation-in-Indian-Equity-Market

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup
python _00_setup_environment.py
```

### 2. Data Pipeline

#### Step 1: Download Data
```bash
python 01_data_download_enhanced.py
# Downloads 10+ years of data for 250+ stocks
# Outputs: data/raw/indian_raw_enhanced.pkl
```

#### Step 2: Preprocess Data
```bash
python 02_data_preprocessing.py
# Applies normalization, technical indicators, missing value handling
# Outputs: data/processed/preprocessed_data.pkl
```

#### Step 3: Create Windowed Dataset
```bash
python 03_data_windowing.py
# Creates sliding windows for time series
# Outputs: data/processed/windowed_data.pkl
```

### 3. Model Training

#### Option A: Standard Pipeline
```bash
python run_pipeline_v4.py
# Trains baseline model
# Generates predictions and evaluation metrics
```

#### Option B: Sector-Aware Pipeline
```bash
python run_sector_pipeline.py
# Trains sector-aware GAT-TFT model
# Includes sector correlation analysis
```

### 4. Evaluation & Results
```bash
python 06_evaluation.py          # Standard metrics (MSE, MAE, Sharpe ratio)
python evaluation_ranking.py     # Ranking and top-k analysis
python visualize_results.py      # Generate visualizations
```

---

## 📈 Key Dependencies

### Core ML Stack
```
torch>=2.0.0              PyTorch deep learning
torch_geometric           Graph neural networks
torch_scatter             Sparse operations
torch_cluster             Graph clustering
torch_spline_conv         Spline convolutions
```

### Data & Science
```
pandas                    Data manipulation
numpy                     Numerical computing
scikit-learn              ML utilities
scipy                     Scientific computing
```

### Data Sources & Utils
```
yfinance                  Stock market data
beautifulsoup4            Web scraping
requests                  HTTP library
networkx                  Graph analysis
tqdm                      Progress bars
```

See `requirements.txt` for complete dependency list.

---

## 📁 Project Structure

```
├── _00_setup_environment.py      # Environment configuration
├── 01_data_download_enhanced.py  # Download 10+ years of NSE data
├── 01_download_*.py              # Alternative download scripts
├── 02_data_preprocessing.py      # Feature engineering & normalization
├── 03_data_windowing.py          # Time series windowing
├── model_*.py                    # Model architectures (v1, v2, v3)
├── sector_graph_model.py         # GAT sector correlation model
├── training*.py                  # Training routines
├── run_pipeline*.py              # End-to-end pipelines
├── run_sector_pipeline.py        # Sector-aware pipeline
├── evaluation*.py                # Evaluation metrics
├── evaluate_topk.py              # Top-K ranking evaluation
├── cross_sectional_loader.py     # Data loading utilities
├── baseline_test.py              # Baseline comparisons
├── visualize_results.py          # Result visualization
│
├── data/
│   ├── raw/                      # Downloaded raw data
│   └── processed/                # Preprocessed data
├── models/                       # Trained model checkpoints
├── results/                      # Evaluation results
├── logs/                         # Training logs
├── Output/                       # Model outputs
└── saved_results/                # Saved metrics & predictions
```

---

## 🔬 Technical Highlights

### 1. **Sector Correlation Analysis**
- Computes dynamic correlation matrices between sectors
- Uses attention mechanisms to weight sector relationships
- Identifies leading/lagging sector relationships

### 2. **Multi-Horizon Forecasting**
- Predicts returns for multiple future time steps
- Produces prediction intervals with uncertainty quantification
- Handles varying forecast horizons

### 3. **Cross-Sectional Information**
- Leverages sector-level patterns for individual stock predictions
- Transfers knowledge from highly correlated sectors
- Improves generalization to out-of-sample data

### 4. **Temporal Dynamics**
- LSTM encoder captures long-term dependencies
- Transformer decoder models future patterns
- Adaptive attention to recent vs. historical data

---

## 📊 Expected Results

### Model Performance Metrics
- **MSE/RMSE**: Root mean squared error on returns
- **MAE**: Mean absolute error
- **Direction Accuracy**: % correct predictions of price direction
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

### Baseline Comparisons
Models include baseline comparisons against:
- LSTM-only models
- Traditional time series (ARIMA)
- Simple moving average strategies

---

## 🎓 Usage Example

```python
# Import utilities
from cross_sectional_loader import load_data
from model_v3_sector_aware import GAT_TFT_Sector
import torch

# Load preprocessed data
X_train, y_train, X_test, y_test = load_data('data/processed/windowed_data.pkl')

# Initialize model
model = GAT_TFT_Sector(
    num_sectors=19,
    num_stocks=250,
    hidden_dim=64,
    num_heads=8,
    forecast_horizon=5
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    predictions = model(X_train)
    loss = torch.nn.functional.mse_loss(predictions, y_train)
    loss.backward()
    optimizer.step()

# Evaluate
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
```

---

## 📝 Configuration

Key configurations in `_00_setup_environment.py`:

```python
CONFIG = {
    'paths': {
        'raw_data_dir': 'data/raw',
        'processed_data_dir': 'data/processed',
        'model_dir': 'models',
        'result_dir': 'results'
    },
    'data': {
        'window_size': 20,           # Days of history
        'forecast_horizon': 5,        # Days to predict
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    },
    'model': {
        'hidden_dim': 64,
        'num_heads': 8,
        'num_layers': 2
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001
    }
}
```

---

## 🔧 Technical Terms Reference

Key financial and technical concepts are documented in `Terms.txt`:
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **ATR**: Average True Range

---

## 📈 Visualization Tools

The project includes comprehensive visualization capabilities:

```bash
# Generate result visualizations
python visualize_results.py
# Produces:
# - Prediction vs Actual plots
# - Sector correlation heatmaps
# - Attention weight visualizations
# - Performance metric dashboards
```

---

## 🤝 Contributing

To improve or extend this project:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Submit pull request with documentation

---

## 📚 References & Inspiration

### Key Papers
- **Graph Attention Networks**: Veličković et al. (2017)
- **Temporal Fusion Transformers**: Lim et al. (2021)
- **Multi-Horizon Time Series Forecasting**: Recent Survey (2023)

### Related Work
- Stock market prediction using attention mechanisms
- Sector correlation networks
- Graph neural networks for finance

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It should not be used for actual trading or investment decisions without thorough backtesting and risk assessment. Past performance does not guarantee future results.

---

## 📧 Support & Issues

For questions, bug reports, or feature requests:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Include error messages and environment details


---

## 🙏 Acknowledgments

- NSE (National Stock Exchange of India) for market data
- PyTorch & PyTorch Geometric teams for excellent frameworks
- Research community for advancing deep learning in finance

---

**Last Updated**: June 2026
**Status**: Active Development
**Python Version**: 3.8+
