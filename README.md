<h2>🩺 Diabetes Patients Readmission Prediction</h2>

A Machine Learning Approach to Predict 30-Day Hospital Readmission Risk

<h3>📌 Introduction</h3>

Hospital readmission among diabetes patients is a major challenge for healthcare systems. Early detection of patients with a high probability of being readmitted enables hospitals to plan interventions, reduce costs, and improve patient outcomes.

This project builds a machine learning model to predict whether a diabetes patient will be readmitted within 30 days using historical patient records, medical attributes, and clinical indicators.

---

<h3>🎯 Project Objective</h3>

To develop a predictive model that classifies patients into:

- **Readmitted < 30 days (high-risk)**  
- **Readmitted > 30 days**  
- **Not readmitted**

**Primary ML Target Variable:**  
`readmitted_30_days` → **1** (readmitted within 30 days), **0** (not readmitted)

---

<h3>🧪 Dataset Summary</h3>

The dataset contains **100,000+ hospital encounters** and includes:

- Patient demographics  
- Diagnoses & procedure codes  
- Laboratory test results  
- Outpatient / inpatient visit history  
- Medication and insulin-related features  
- Admission & discharge information  

**Data Cleaning Performed:**

✔ Handling missing values  
✔ Fixing inconsistent categories  
✔ Reducing noise in diagnosis codes  
✔ Creating new engineered features for better model accuracy  

---

<h3>📊 Exploratory Data Analysis (Key Findings)</h3>

**🧱 Correlation Heatmap**  
Most numerical features show low correlation → dataset is primarily categorical.

**👵 Age Distribution**  
Majority of patients fall in the **50–80 age group**, highlighting higher chronic diabetes prevalence.

**🔁 Readmission Distribution**  
- “Not readmitted” → majority class  
- “Readmitted < 30 days” → minority class requiring class-imbalance handling  

**⭐ Important Predictors:**

- Number of prior inpatient visits  
- Number of outpatient visits  
- Number of diagnoses  
- Change in diabetes medication  
- Insulin dosage  
- Time spent in the hospital  

---

<h3>📉 Model Evaluation (Charts)</h3>

Generated evaluation charts include:

- Confusion matrix  
- ROC curve  
- F1-score comparison  
- Model accuracy comparison  
- Random Forest feature importance  

These visualizations help interpret model performance and identify improvement areas.

---

<h3>🛠️ Feature Engineering</h3>

Key transformations performed:

✔ Label encoding & one-hot encoding  
✔ SMOTE oversampling for minority class  
✔ Scaling numerical variables  
✔ Grouping diagnosis codes  
✔ Creating composite features:
  - Total number of visits  
  - Diabetes medication change flags  
  - Chronic illness indicators  

**Final processed dataset:**  
`feature_engineered_diabetes.csv`

---

<h3>🤖 Machine Learning Models</h3>

The following models were trained and compared:

**Logistic Regression**
  <img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/00a5f38b-f5e6-443d-adf3-b9304e516bf7" /> 
**Random Forest Classifier**
  <img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/034824bd-487e-4bd3-b678-4f87882fea2e" />
**XGBoost Classifier**
  <img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/183554d3-c5cd-4f8c-b65d-bf1a936795b3" />

---

<h3>⭐ Best Performing Models</h3>

- **XGBoost** → highest accuracy  
- **Random Forest** → most stable & interpretable
  <img width="526" height="120" alt="Image" src="https://github.com/user-attachments/assets/e60f8807-fac8-4ca0-a35f-2b2785d099d5" />

Both models performed well in identifying high-risk patients.

---

<h3>📈 Results Summary</h3>

- **Accuracy:** ~85%  
- **ROC-AUC:** ~0.87  
- **F1-Score:** strong for majority class, significantly improved for minority class after SMOTE  
- **Interpretability:** insights extracted from feature importance  

The model delivers meaningful predictions that can support hospital decision-making and help reduce readmission rates.

