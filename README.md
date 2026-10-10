# AI Loan Approval Assistant

A Streamlit web app that estimates whether a loan application would be approved, explains the decision with SHAP, and gives administrators a dashboard of model quality and fairness.

> **Disclaimer:** This is a decision-support estimate produced by a machine learning model trained on a synthetic public dataset. It is not an actual bank decision.

## Features

- **Login and registration** with two roles (admin, officer). Passwords are salted PBKDF2-SHA256 hashes stored in SQLite.
- **Assess page:** enter an application and get an approval probability, a decision, and a SHAP chart of the top factors behind it.
- **Unusual-application warning:** an Isolation Forest flags applications that look very different from the training data, so the result can be treated as unreliable.
- **History page:** officers see their own predictions, admins see everyone's. Filters and CSV download included.
- **Admin dashboard:** KPI cards, model comparison, test metrics, gender fairness audit, dataset insights and recent activity.

## Tech stack

Python 3.11, Streamlit, scikit-learn, XGBoost, SHAP, Plotly, pandas, SQLite, pytest.

## Dataset

Kaggle "Loan Approval Classification Dataset" (taweilo), 45,000 rows, saved as `data/loan_data.csv`. It is synthetic, so real-world accuracy would be lower.

## Model results

Three models (Logistic Regression, Random Forest, XGBoost) were compared with 3-fold cross-validated ROC-AUC on the training set. The winner, XGBoost, was evaluated once on a held-out 20% test set:

| Metric | Value |
|---|---|
| Accuracy | 0.938 |
| Precision | 0.901 |
| Recall | 0.810 |
| F1 | 0.853 |
| ROC-AUC | 0.980 |

**Fairness:** gender is never a model input. It is used only to audit predicted approval rates on the test set (female 19.6%, male 20.2%).

## Setup (Windows, PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset at `data\loan_data.csv`, then:

```powershell
python scripts/train.py            # trains the loan model, saves models\best_pipeline.joblib
python scripts/train_anomaly.py    # trains the unusual-application detector
python scripts/init_db.py          # creates data\app.db and the first admin user
streamlit run app.py
```

Open http://localhost:8501 and log in with the admin account you created.

## Tests

```powershell
python -m pytest -q
```

## Project structure

```
app.py                  Streamlit entry point, login gate and navigation
config.py               Paths, column names and constants
src/
  db.py                 SQLite connection and schema
  auth.py               Registration, login, password hashing
  data.py               Loading, cleaning, train/test split
  pipeline.py           Preprocessing and candidate models
  predict.py            Scoring and saving predictions
  explain.py            SHAP explanations
  anomaly.py            Unusual-application detector
  analytics.py          Data helpers for history and dashboard
views/                  Streamlit pages (login, predict, history, dashboard)
scripts/                train.py, train_anomaly.py, init_db.py
tests/                  pytest suites
.streamlit/config.toml  Dark theme
```

## Limitations

- Trained on synthetic data; the learned patterns do not reflect real lending policy.
- Recall is about 0.81, so some creditworthy applicants would be rejected.
- Only gender is audited for fairness; other attributes are not.
- Not intended for real credit decisions.