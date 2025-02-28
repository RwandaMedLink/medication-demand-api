# Medication Demand Prediction API

## 🚀 Overview

The **Medication Demand Prediction API** is an AI-powered system designed to help pharmacies optimize their inventory by predicting future medication demand. Using historical sales data and external factors such as promotions, holidays, and competition, this API provides insights to prevent stockouts and overstocking.

This project is part of **RwandaMedLink**, a platform aimed at improving pharmaceutical supply chain efficiency in Rwanda.

---

## 🎯 Features

✅ **Predict Future Demand**: Estimates the number of units required per future time period.  
✅ **Stock Optimization**: Generates tailored restocking recommendations for each pharmacy.  
✅ **Sales Impact Analysis**: Considers promotions, holidays, and seasonality.  
✅ **Proactive Alerts**: Notifies pharmacies about potential stockouts or overstock situations.

---

## 🔬 AI Model & Methodology

We evaluated three machine learning models:

- **Logistic Regression (LR)**
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**

After testing, we selected the **most performant model** based on accuracy, precision, and real-world applicability.

### **Model Inputs**

- 📊 **Historical sales data** (hourly & daily)
- 🏷 **ATC code** of medications
- 🎯 **Promotions & marketing campaigns**
- 📆 **Holidays & school closures**
- 🏪 **Competition metrics** (distance, opening dates)

### **Model Outputs**

- 📈 **Predicted medication demand**
- 📦 **Stock replenishment recommendations**
- ⚠ **Stockout & overstock alerts**

---

## 🏗 API Endpoints

| Method | Endpoint           | Description                                         |
| ------ | ------------------ | --------------------------------------------------- |
| `POST` | `/predict`         | Predicts medication demand based on input data.     |
| `GET`  | `/recommendations` | Provides restocking recommendations.                |
| `GET`  | `/alerts`          | Retrieves alerts for stockouts and overstock risks. |

---

## 🔧 Installation & Setup

```sh
git clone https://github.com/Gihozo23/medication-demand-api.git
cd medication-demand-api
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000

```
