<img width="994" height="741" alt="image" src="https://github.com/user-attachments/assets/7658a033-9c59-473d-829c-7fe0b4a80d4c" />


# Machine Learning Powered Intelligent Credit Card Fraud Detection

This project focuses on building a complete, end‑to‑end machine learning pipeline to detect fraudulent credit card transactions using anonymized numerical features (V1–V28) along with the transaction Amount. The goal is to create a practical, dependable, and production‑ready fraud detection system that includes data processing, model training, evaluation, saving, and a user‑friendly Streamlit interface.

---

## 📌 Project Overview

Credit card fraud is a rare but high‑impact problem. Most transactions are genuine, making fraud detection a highly imbalanced classification task. This project walks through the entire workflow of building an intelligent fraud detection model—from raw data all the way to a deployed UI.

The model predicts whether a given transaction is **fraudulent (1)** or **legitimate (0)** using machine learning, with a strong emphasis on proper data handling, class imbalance solutions, and clean deployment.

---

## 🚀 Project Workflow

Below is the exact workflow followed in this project:

### **🟦 Step 1: Data Ingestion**

* Load the raw credit card dataset (usually from Kaggle).
* Inspect column structure (V1–V28 + Amount + Class).
* Store raw files safely without modification.

### **🟥 Step 2: Exploratory Data Analysis (EDA)**

* Understand distribution of features.
* Analyze correlations and detect any anomalies.
* Review imbalance in the target variable.
* Visualize transaction Amount behavior for fraud vs non‑fraud.

### **🟪 Step 3: Preprocessing**

* Scale numerical values (StandardScaler).
* Optional PCA if needed (depends on experimentation).
* Clean or transform Amount/Time appropriately.

### **🟧 Step 4: Train‑Test Split**

* Use stratified splitting to preserve fraud ratio.
* Prevent leakage by splitting **before** any resampling.

### **🟨 Step 5: Handle Class Imbalance**

* Fraud cases are extremely rare, so imbalance handling is critical.
* Techniques evaluated:

  * Class weights
  * LightGBM’s built‑in `is_unbalance` or `scale_pos_weight`
  * Oversampling (SMOTE)
  * Undersampling

### **🟩 Step 6: Model Building**

* Multiple models experimented with:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * **LightGBM** (final choice)
* LightGBM performed best due to:

  * Great performance on high‑dimensional numerical data
  * Fast training
  * Good handling of imbalance

### **🟫 Step 7: Final Model Selection**

* Evaluate models based on:

  * Precision, Recall, F1
  * ROC‑AUC
  * PR‑AUC (most important for imbalance)
* Select the best performing model (LightGBM).

### **⬛ Step 8: Saving the Final Model**

* A full pipeline was created using scikit‑learn:

  * Preprocessing (Scaler/PCA)
  * LightGBM model
* Saved using `joblib` for deployment:

```
fraud_detection_pipeline.pkl
```

### **🟫 Step 9: Build Streamlit UI**

* A clean user interface was created to allow:

  * Manual input of V1–V28 + Amount
  * Real‑time fraud prediction
  * Probability display for transparency
* The UI loads the saved pipeline and performs inference instantly.

---

##  Machine Learning Model

**Model Used:** LightGBM Classifier

**Why LightGBM?**

* Handles numerical, high‑dimensional data well.
* Fast and scalable.
* Works better than XGBoost for heavily imbalanced datasets with fewer hyperparameters to tune.

**Key Features Used:**

* V1 – V28 (PCA‑transformed anonymized features)
* Amount

**Target:** `Class` → 0 (Legit) / 1 (Fraud)

---

## 📊 Evaluation

Metrics considered:

* **Precision** (How many flagged transactions were actually fraud?)
* **Recall** (How many fraud transactions did the model catch?)
* **F1 Score** (Balance of precision & recall)
* **ROC-AUC**
* **PR-AUC** (best metric for this dataset)

The final LightGBM model achieved strong performance, especially in PR‑AUC and recall, which are critical for real‑world fraud detection.

---

## 💾 Model Saving & Loading

The model is saved as a pipeline for clean inference:

```python
import joblib
pipeline = joblib.load("fraud_detection_pipeline.pkl")
prediction = pipeline.predict(input_data)
```

This ensures preprocessing and model prediction always match the training setup.

---

## 💻 Streamlit App Features

* Clean UI for manually entering transaction details.
* One‑click fraud prediction.
* Probability output to understand model confidence.
* Ready for deployment on Streamlit Cloud.

Run locally:

```bash
streamlit run streamlit_app.py
```

---



## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit‑learn
* LightGBM
* Imbalanced‑Learn
* Streamlit
* Matplotlib / Seaborn

---

##  How to Use this Project

1. Clone the repository.
2. Install required libraries.
3. Run training notebook if you want to retrain.
4. Launch UI using Streamlit.
5. Input feature values and get predictions instantly.

---

##  Future Improvements

* Add SHAP explainability in UI.
* Introduce FastAPI service for real‑time API predictions.
* Add batch prediction support.
* Enable automatic retraining on new data.

---

## Acknowledgements

* Dataset sourced from Kaggle’s Credit Card Fraud Detection dataset.
* Inspired by real‑world transaction monitoring systems.

---

If you find this project useful, consider starring ⭐ the repository or contributing improvements!

