# Privacy-Preserving Credit Scoring Using Federated Learning with Differential Privacy

## Project Overview

In the modern financial industry, credit scoring plays a crucial role in evaluating the creditworthiness of customers. Traditional credit scoring systems require centralized data collection from multiple banks, which creates serious concerns regarding data privacy, security risks, regulatory compliance, and customer trust.

This project solves that problem using Federated Learning (FL) with Differential Privacy (DP), where multiple banks collaboratively train a machine learning model without sharing raw customer data.

Only model parameters are exchanged, ensuring strong privacy preservation while maintaining high prediction performance.

This project is implemented as a professional interactive dashboard using Streamlit, making it suitable for Final Year Project presentation and demonstration.

---

## Objectives

* Build a privacy-preserving credit scoring system
* Avoid centralized sharing of sensitive customer data
* Compare multiple learning approaches
* Apply Differential Privacy for formal privacy guarantees
* Provide real-time visualization of FL training
* Predict customer credit scores (CIBIL-style)

---

## Key Features

### Federated Learning

Train models across multiple banks without moving raw customer data.

### Differential Privacy

Protect sensitive customer information using gradient clipping and Gaussian noise.

### Real-Time Dashboard

Professional Streamlit dashboard for monitoring training and privacy metrics.

### Performance Comparison

Compare:

* Centralized Learning
* FedAvg
* FedAvg + DP
* FedProx + DP
* Local-only Baselines

### Credit Score Predictor

Predict customer credit score using trained model.

### Privacy Analysis

Visualize epsilon budget and privacy-accuracy tradeoff.

---

## Participating Banks

The system simulates training across 5 banks:

* SBI
* HDFC
* ICICI
* Axis Bank
* Punjab National Bank (PNB)

Each bank contains non-IID customer financial datasets.

---

## Technology Stack

| Category             | Technology         |
| -------------------- | ------------------ |
| Frontend             | Streamlit          |
| Visualization        | Plotly             |
| Machine Learning     | PyTorch            |
| Federated Learning   | Custom FL + Flower |
| Differential Privacy | Custom DP + Opacus |
| Data Processing      | Pandas, NumPy      |
| Backend Logic        | Python             |

---

## Project Structure

project/

├── app.py

├── src/

│   ├── data/

│   │   └── data_generator.py

│   ├── models/

│   │   └── model.py

│   ├── federated/

│   │   ├── fl_engine.py

│   │   ├── baseline.py

│   │   ├── comparison.py

│   │   └── flower_simulation.py

│   ├── privacy/

│   │   └── dp_custom.py

│   ├── ui/

│   │   ├── login.py

│   │   ├── sidebar.py

│   │   ├── overview.py

│   │   ├── fl_training.py

│   │   ├── baseline_comparison.py

│   │   ├── performance_comparison.py

│   │   └── privacy_analysis.py

│   └── utils/

│       ├── plots.py

│       └── fl_logger.py

└── requirements.txt

---

## Installation

### Step 1: Clone Project

git clone your-repository-link

cd project-folder

---

### Step 2: Install Requirements

pip install -r requirements.txt

---

### Step 3: Run Application

streamlit run app.py

---

### Step 4: Login Credentials

Username: admin

Password: admin123

---

## Dashboard Modules

### 1. Overview

Displays:

* Project architecture
* FL workflow
* Privacy concept
* System metrics

---

### 2. Data Explorer

Shows:

* Bank-wise data distribution
* Non-IID dataset behavior
* Feature visualization
* Raw dataset preview

---

### 3. FL Training

Used for:

* Training FL models
* Live round-by-round visualization
* Accuracy and AUC monitoring
* Global model evaluation

---

### 4. Baseline Comparison

Compare:

* Centralized Learning (Upper Bound)
* Local-only Models (Lower Bound)
* FL + DP Model (Main Approach)

---

### 5. Performance Comparison

Compare all four major modes:

| Method       | Privacy |
| ------------ | ------- |
| Centralized  | No      |
| FedAvg       | No      |
| FedAvg + DP  | Yes     |
| FedProx + DP | Yes     |

---

### 6. Privacy Analysis

Displays:

* Epsilon calculation
* Privacy budget usage
* DP strength
* Accuracy vs Privacy tradeoff

---

### 7. Credit Predictor

Predicts:

* Creditworthiness
* CIBIL-style score
* Loan approval suggestion

based on customer details.

---

## Privacy Guarantees

### Without FL

Banks must share raw customer data.

### With FL + DP

Banks only share model updates.

### Differential Privacy

Noise is added to gradients to prevent:

* Data leakage
* Membership inference attacks
* Sensitive information exposure

Target privacy guarantee:

Epsilon < 3

which is considered strong privacy for banking applications.

---

## Expected Results

| Method       | Accuracy | Privacy | Data Shared   |
| ------------ | -------- | ------- | ------------- |
| Centralized  | ~89%     | None    | Raw Data      |
| FedAvg       | ~87%     | None    | Model Weights |
| FedAvg + DP  | ~84%     | Strong  | Model Weights |
| FedProx + DP | ~85%     | Strong  | Model Weights |
| Local-only   | ~75–80%  | Full    | No Sharing    |

---

## Final Year Project Advantages

This project demonstrates:

* Real-world banking use case
* AI + ML application
* Federated Learning
* Differential Privacy
* Production-ready dashboard
* Research-oriented implementation
* Strong presentation value

This makes it highly suitable for:

* Final Year Project
* Placement Interviews
* Research Paper Extension
* Higher Studies Portfolio

---

## Key Achievements

* Privacy-preserving ML system
* Professional dashboard UI
* Real-time training visualization
* Strong differential privacy guarantees
* Multiple model comparison
* Credit score prediction system
* Modular production-ready code
* Presentation-ready final year project

---



