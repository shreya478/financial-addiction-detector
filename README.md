# Financial Addiction Detection System
**A privacy-preserving machine learning platform for detecting and mitigating high-risk financial behaviors.**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

## Overview
Compulsive financial behavior, such as problem gambling or erratic trading, can lead to devastating personal consequences. This project provides a secure, high-fidelity system to detect financial addiction across 5000+ simulated users. By leveraging DSM-5 based proxy labels, Differential Privacy (DP-SGD), and real-time inference, the system categorizes users into distinct risk tiers while ensuring strict adherence to data privacy standards and defending against adversarial manipulation.

## Architecture

```text
+-------------------+      +-------------------+      +---------------------+
|                   |      |                   |      |                     |
|  Data Generator   +----->+  Labeling Engine  +----->+ Feature Engineering |
|                   |      |                   |      |                     |
+-------------------+      +-------------------+      +----------+----------+
                                                                 |
                                                                 v
+---------------------------------------------------------------------------------+
|                                 ML Pipeline                                     |
|                                                                                 |
|  +--------------------+  +--------------------+  +---------------------------+  |
|  | DBSCAN / IsoForest |  | DP-SGD Classifier  |  |      SHAP Explainer       |  |
|  |    Clustering      |  |  (Risk Tiers 0-3)  |  | (Feature Interpretability)|  |
|  +--------------------+  +--------------------+  +---------------------------+  |
|                                                                                 |
|                         +---------------------------+                           |
|                         |   Adversarial Detector    |                           |
|                         +---------------------------+                           |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
                            +-----------------------+
                            |                       |
                            |    FastAPI Backend    |
                            |                       |
                            +-----------+-----------+
                                        |
                                        v
                            +-----------------------+
                            |                       |
                            |  Streamlit Dashboard  |
                            |                       |
                            +-----------------------+
```

## Key Features
*   **Behavioral Baseline Generation:** Utilizes DBSCAN clustering and Isolation Forests to establish unsupervised behavioral baselines from raw transaction data.
*   **Differentially Private Training:** Employs Opacus (DP-SGD) to train a PyTorch neural network, protecting sensitive user transaction data with mathematical privacy guarantees.
*   **Imbalance Handling & Explainability:** Applies SMOTE to manage class imbalance for high-risk minority classes. Integrates SHAP to provide real-time, plain-English feature importance metrics.
*   **Adversarial Robustness:** Incorporates an adversarial detection layer to identify and mitigate adversarial input perturbations designed to fool the risk classifier.
*   **Platform Accountability:** Includes a Dark Pattern Scorer that calculates the Pearson correlation between platform nudges (e.g., post-loss notifications) and subsequent user risk escalation.
*   **Real-time Intervention Engine:** Classifies transaction sequences dynamically into Casual (0), Frequent (1), Compulsive (2), and Crisis (3) states via state-machine logic.

## Model Performance

| Metric | Value |
| :--- | :--- |
| Precision | 91%+ |
| Recall | 89%+ |
| F1-Score | 90%+ |
| AUC-ROC | 0.94+ |

## Project Structure
```text
financial-addiction-detector/
├── api/
│   ├── auth.py              # JWT Authentication
│   ├── deps.py              # ML Model caching and rate limiting
│   ├── main.py              # FastAPI application entry point
│   ├── routes.py            # API endpoints
│   └── schemas.py           # Pydantic validation models
├── data/                    # Generated datasets and metrics logs
├── frontend/
│   └── app.py               # Streamlit interactive dashboard
├── models/
│   ├── adversarial_detector.py
│   ├── clustering_analysis.py
│   ├── dark_pattern_scorer.py
│   ├── dp_classifier.py
│   ├── explainability.py
│   ├── final_dp_model.pth
│   ├── schema.sql
│   └── state_machine.py
├── scripts/                 # Validation and testing scripts
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/financial-addiction-detector.git
   cd financial-addiction-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Launch the Streamlit frontend (in a new terminal):**
   ```bash
   streamlit run frontend/app.py
   ```

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/risk-score/{user_id}` | GET | Retrieves the predicted risk tier (0-3) for a specific user |
| `/state/{user_id}` | GET | Returns the behavioral state and DSM-5 proxy score |
| `/explain/{user_id}` | GET | Generates SHAP-based plain-English explanations for risk classification |
| `/platform-report` | GET | Fetches regulator dashboard leaderboard (dark pattern analysis) |
| `/alert/{user_id}` | POST | Evaluates user state and triggers intervention alerts if in Crisis |
| `/score/predict` | POST | Real-time inference on a live payload of transactions |

## Tech Stack
*   **Machine Learning:** PyTorch, Opacus, Scikit-learn, SMOTE, SHAP
*   **Backend:** FastAPI, Pydantic, Uvicorn
*   **Frontend:** Streamlit, Plotly
*   **Data & Storage:** Pandas, NumPy, PostgreSQL (Schema definition)

## Future Work
*   Integrate real-world, anonymized transaction datasets to validate synthetic baselines.
*   Implement LLM-based, personalized intervention messaging tailored to specific user vulnerabilities.
*   Develop a mobile application with push notification capabilities for real-time crisis alerts.

*Note: The `api/auth.py` file uses dummy credentials for demonstration purposes. Do not use this authentication setup in production.*