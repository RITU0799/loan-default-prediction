# Loan Default Risk Prediction - Web Application

An **Intelligent, Interactive Dashboard** for predicting loan default risk using **Gradient Boosting Models (GBM)** and **Visual Analytics**.  
This project combines **Machine Learning**, **Streamlit**, and **Tableau** to deliver both **predictive modeling** and **business insights** in a single application.

---

## 🚀 Features

- **📄 Customer Application Form** – Capture borrower details such as loan amount, income, credit grade, term, etc.
- **🎯 Risk Scoring** – Leverages a trained **GBM pipeline** to classify applications into **High Risk** or **Low Risk**.
- **📊 Visualization Insights** – Includes loan amount vs. income trends, feature relationships, and pairwise correlations.
- **🔍 Interactive Visuals** – Scatter plots, histograms, box plots, and categorical breakdowns with dynamic filtering.
- **📈 Embedded Tableau Dashboard** – Advanced **BI analytics** for deeper exploration of loan patterns.
- **📝 Prediction Logging** – Automatically stores application data and risk results into `prediction_logs.csv` for audit and analysis.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://loan-default-dashboard.streamlit.app/)  
- **Machine Learning:** `scikit-learn`, GradientBoostingClassifier  
- **Visualization:** `Plotly Express`, Tableau (embedded)  
- **Data Handling:** `Pandas`, `NumPy`  
- **Model Storage:** Git LFS for large ML models  
- **Version Control:** Git, GitHub

---

## 📌 Future Enhancements

- 🌐 Deploy on Render/Streamlit Cloud with persistent storage  
- 🔍 Integrate real-time credit bureau API for live credit data  
- 🧠 Add SHAP Explainability to interpret model predictions  
- 🔒 Implement user authentication for secure data access

---

## 📦 Installation & Setup

```bash
git clone https://github.com/RITU0799/loan-default-prediction.git
cd loan-default-prediction

git init
git lfs install
git lfs track "models/gbm_pipeline.pkl" "models/model_columns.pkl"
git lfs track "data/lending_club_loan_two.csv" "data/loan_data_cleaned.csv"
git lfs track "notebooks/Python_Visualization.ipynb"

pip install -r requirements.txt

streamlit run app.py
App will be available at: http://localhost:8501

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Author: Ritul Gaikwad
LinkedIn: https://www.linkedin.com/in/ritul-gaikwad-9286b5182/
