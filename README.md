<div align="center">

# Fraud Detection System

**Production-grade ML pipeline for real-time financial fraud detection**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Online-brightgreen?style=flat-square)](https://fraud-detection-system-73mm.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-orange?style=flat-square)](tests/)

[Live App](https://fraud-detection-system-73mm.onrender.com) · [Features](#key-features) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Performance](#model-performance)

</div>

---

## Overview

An end-to-end machine learning system that detects fraudulent mobile money transactions in real time. Trained on 6.3M+ PaySim transactions with a 774:1 class imbalance, the system achieves **99.83% PR-AUC** with sub-100ms prediction latency — and an estimated **$6.07M net savings** in a simulated business scenario.

| | |
|---|---|
| **Domain** | FinTech / Risk Management |
| **Dataset** | PaySim — 6,362,620 synthetic mobile money transactions |
| **Fraud Rate** | 0.13% (8,213 fraud cases, 774:1 imbalance ratio) |
| **Champion Model** | Random Forest — 99.83% PR-AUC |
| **Latency** | < 100ms per prediction |
| **Business Impact** | $6.07M net savings, 98.6% ROI |

---

## Key Features

**ML Pipeline**
- Automated, modular workflow: ingestion → validation → feature engineering → training → evaluation → deployment
- 15+ engineered features including balance error signals, zero-balance flags, merchant patterns, and amount-to-balance ratios
- Multi-model training: Random Forest, XGBoost, LightGBM, and Logistic Regression — compared head-to-head on PR-AUC
- Imbalanced data handling via class weight tuning and threshold optimization

**Web Application**
- Real-time transaction scoring via a Flask REST API
- Clean, responsive UI with confidence scores and human-readable explanations
- Mobile-friendly dark theme

**Engineering**
- Fully containerized with Docker and Docker Compose
- CI/CD via GitHub Actions → Render deployment
- pytest test suite with unit and integration coverage
- Structured logging, custom exception classes, and full type annotations

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / AI | scikit-learn, XGBoost, LightGBM |
| Web | Flask, Jinja2 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Testing | pytest |
| Deployment | Docker, Render |
| Version Control | Git, GitHub |

---

## Project Structure

```
fraud-detection-system/
├── notebook/
│   ├── data/paysim_fraud_data.csv        # Raw dataset (6.3M+ transactions)
│   ├── 01_PaySim_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training_Evaluation.ipynb
│
├── src/
│   ├── exception.py                       # Custom exception classes
│   ├── logger.py                          # Logging configuration
│   ├── utils.py                           # Shared utilities
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   └── pipeline/
│       ├── train_pipeline.py
│       └── predict_pipeline.py
│
├── artifacts/                             # Saved models, preprocessors, plots
├── templates/                             # Flask HTML templates
├── tests/
│   ├── unit/
│   └── integration/
├── dashboard/
│   └── Fraud_Operations_Dashboard.twbx   # Tableau executive dashboard
├── application.py                         # Flask entry point
├── config.yaml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- Git
- Docker (optional)

### Installation

```bash
# Clone the repo
git clone https://github.com/AyushPaderiya/fraud-detection-system.git
cd fraud-detection-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .                # Optional: install as importable package
```

### Train the Pipeline

```bash
python src/pipeline/train_pipeline.py
```

This will load the raw data, validate it, engineer features, train all four models, select the best by PR-AUC, and save artifacts to `artifacts/`.

Expected output:
```
[INFO] Data ingestion completed: 6,362,620 transactions loaded
[INFO] Train: 3,817,572 | Val: 1,272,524 | Test: 1,272,524
[INFO] Feature engineering completed: 15 features
[INFO] Champion Model: Random Forest (PR-AUC: 0.9983)
[INFO] Model saved to artifacts/model.pkl
```

### Run the Web App

```bash
python application.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) — the transaction scanner is at `/predict`.

---

## Usage

### Programmatic Prediction

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    step=1,
    type="TRANSFER",
    amount=50000.00,
    nameOrig="C123456789",
    oldbalanceOrg=60000.00,
    newbalanceOrig=10000.00,
    nameDest="C987654321",
    oldbalanceDest=0.00,
    newbalanceDest=50000.00,
    isFlaggedFraud=0
)

pipeline = PredictPipeline()
result = pipeline.predict(data.to_dataframe())
print(result)  # "FRAUD" or "LEGITIMATE"
```

---

## Dataset

The model is trained on the [PaySim1 dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic simulation of mobile money transactions modeled on real anonymized data from a mobile money service in Africa.

### Raw Features

| Feature | Description |
|---|---|
| `step` | Hour-resolution time step (30-day simulation) |
| `type` | Transaction type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Origin account balance before/after |
| `oldbalanceDest` / `newbalanceDest` | Destination account balance before/after |
| `isFlaggedFraud` | Rule-based flag: transfers > 200K |
| `isFraud` | Ground truth label |

### Engineered Features (15+)

```python
# Accounting inconsistencies — powerful fraud signals
balance_error_orig = oldbalanceOrg - newbalanceOrig - amount
balance_error_dest = newbalanceDest - oldbalanceDest - amount

# Risk flags
is_zero_balance_orig  = (oldbalanceOrg == 0)
is_zero_balance_dest  = (oldbalanceDest == 0)
is_merchant_dest      = nameDest.startswith('M')

# Normalized transaction size
amount_to_balance_ratio = amount / (oldbalanceOrg + 1)
```

---

## Model Performance

### Champion: Random Forest

| Metric | Score |
|---|---|
| PR-AUC | **0.9983** |
| ROC-AUC | 0.9999 |
| F1-Score | 0.9980 |
| Precision | 1.0000 |
| Recall | 0.9968 |

### Confusion Matrix (Validation Set)

```
                  Predicted Legit   Predicted Fraud
Actual Legit         1,270,000            0
Actual Fraud                 5        1,519
```

Zero false positives. Five missed fraud cases out of 1,524.

### Model Comparison

| Model | PR-AUC | ROC-AUC | F1 | Training Time |
|---|---|---|---|---|
| **Random Forest** ⭐ | **0.9983** | **0.9999** | **0.9980** | ~6 min |
| XGBoost | 0.9920 | 0.9995 | 0.9850 | ~12 min |
| LightGBM | 0.9910 | 0.9993 | 0.9830 | ~8 min |
| Logistic Regression | 0.8520 | 0.9750 | 0.7230 | ~2 min |

Random Forest was selected for its superior PR-AUC (the right metric for highly imbalanced data), zero false positives at the optimal threshold, and fast inference (< 10ms per call).

### Business Impact

| Metric | Value |
|---|---|
| Fraud prevented | $6.13M |
| Fraud missed | $25K |
| False positive cost | $0 |
| Net savings | $6.07M |
| ROI | 98.6% |
| Inference cost | < $0.01 / transaction |

---

## Screenshots

### Homepage
![Homepage](assets/homepage_screenshot.png)

### Transaction Scanner
![Transaction Scanner](assets/transaction_scanner.png)

### Legitimate Transaction
![Legitimate Result](assets/predict_screenshot.png)

### Fraud Alert
![Fraud Alert](assets/fraud_alert.png)

### Tableau Dashboard
An executive-level Fraud Risk Operations Command Center with KPI tiles, 30-day trend analysis, an hour-of-day fraud heatmap, and filters by date range, transaction type, and risk tier.

![Tableau Dashboard](assets/Dashboard.png)

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Homepage |
| GET | `/predict` | Transaction scanner form |
| POST | `/predict` | Submit a transaction for scoring |

### POST `/predict` — Request

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 50000.00,
  "nameOrig": "C123456789",
  "oldbalanceOrg": 60000.00,
  "newbalanceOrig": 10000.00,
  "nameDest": "C987654321",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 50000.00,
  "isFlaggedFraud": 0
}
```

### POST `/predict` — Response

An HTML page containing the prediction result and confidence score.

### cURL Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "step=1" -F "type=TRANSFER" -F "amount=50000" \
  -F "nameOrig=C123456789" -F "oldbalanceOrg=60000" \
  -F "newbalanceOrig=10000" -F "nameDest=C987654321" \
  -F "oldbalanceDest=0" -F "newbalanceDest=50000" \
  -F "isFlaggedFraud=0"
```

---

## Docker Deployment

```bash
# Build
docker build -t fraud-detection-system:latest .

# Run
docker run -d -p 5000:5000 --name fraud-app fraud-detection-system:latest
```

App available at [http://localhost:5000](http://localhost:5000).

### CI/CD Pipeline

Pushes to `main` trigger a GitHub Actions workflow that:
1. Provisions an Ubuntu runner and installs dependencies
2. Runs the full `pytest` suite
3. On success, fires a Render deploy hook to build and deploy the Docker image

To trigger a deployment manually:
```bash
git add . && git commit -m "chore: trigger deploy" && git push origin main
```

Monitor progress in the **Actions** tab on GitHub, then verify the live app on [Render](https://render.com).

---

## Testing

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=src

# Run a specific module
pytest tests/unit/test_data_ingestion.py -v

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in your browser
```

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make changes — follow PEP 8, add tests, update docs as needed
3. Run `pytest tests/` to confirm everything passes
4. Push and open a Pull Request

Please use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages and be constructive in code review.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Ayush Paderiya** — Data Analyst & ML Engineer

📧 paderiyaayush@gmail.com · [GitHub](https://github.com/AyushPaderiya) · [Issues](https://github.com/AyushPaderiya/fraud-detection-system/issues)

---

## Acknowledgments

- **PaySim dataset** by Edgar Alonso Lopez-Rojas
- scikit-learn, XGBoost, LightGBM, Flask, and the broader Python ML ecosystem
- Kaggle community for dataset hosting and discussion

---

<div align="center">

If this project was useful to you, a ⭐ on GitHub goes a long way.

</div>