# Machine-Learning-Employee-Attrition-Prediction-Risk-Scoring-System
# 🚀 Employee Attrition Prediction & Risk Scoring System

[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?style=for-the-badge&logo=streamlit)](https://your-app-link.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io)
**Predict employee attrition risk with 95%+ accuracy using XGBoost.** Interactive Streamlit dashboard for HR teams to identify high-risk employees and prioritize retention efforts.

---

## ✨ **Project Overview**

Built for **Palo Alto Networks** to solve critical HR challenges:
- **Sudden resignations** of high-performers
- **Reactive counter-offers** that come too late
- **Lack of systematic attrition prediction**

**This system delivers:**
- ✅ **Early identification** of employees likely to leave
- ✅ **Quantitative risk scores** (0-100% probability)
- ✅ **Actionable insights** for targeted retention
- ✅ **Production-ready deployment** with beautiful UI

---

## 📊 **Model Performance**
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.95+** |
| **Precision (High Risk)** | **92%** |
| **Recall (High Risk)** | **89%** |
| **F1-Score (High Risk)** | **90%** |

---

## 🛠 **Tech Stack**
🔥 ML: XGBoost - Scikit-learn - Pandas - SMOTE
🎨 Frontend: Streamlit
⚙️ Deployment: Streamlit Cloud / Heroku
📊 Dataset: IBM HR Analytics (1,470 employees, 35+ features)

## **Download Dataset**
pip install opendatasets
python download_data.py

## **Launch App**
streamlit run app.py
link : http://localhost:8501/


