# Machine-Learning-Powered-Intelligent-Credit-Card-Fraud-Detection

🪙 Credit Card Fraud Detection using Machine Learning
🔍 Overview

This project detects fraudulent credit card transactions using Machine Learning.
It leverages LightGBM, SMOTE (for class imbalance handling), and a Streamlit web app for real-time predictions.

🚀 Built end-to-end — from data preprocessing and model training to deployment.


-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------



🧠 Tech Stack

Python 3.12

LightGBM – high-performance gradient boosting

SMOTE – Synthetic Minority Oversampling Technique

scikit-learn – scaling, evaluation, and preprocessing

Streamlit – web app interface

Joblib – model serialization



-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------



⚙️ Installation & Setup


1️⃣ Clone the repository
git clone https://github.com/<your-username>/credit-fraud-detection.git
cd credit-fraud-detection


2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate     # (Mac/Linux)
venv\Scripts\activate        # (Windows)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model
python train_model.py

5️⃣ Run the Streamlit app
streamlit run app.py


-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------





🧩 Workflow

Data Loading → Load creditcard.csv

Preprocessing → Standardize data using StandardScaler

Imbalance Handling → Apply SMOTE to balance fraud/non-fraud classes

Model Training → Train LightGBM classifier

Evaluation → View precision, recall, and F1-score

Deployment → Predict transactions using the Streamlit dashboard



-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------





🧾 Sample Output

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.99      0.99      0.99     56864

    accuracy                           0.99    113728
   macro avg       0.99      0.99      0.99    113728
weighted avg       0.99      0.99      0.99    113728


-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------






🖥️ App Preview

Your Streamlit interface allows users to:

Enter transaction details manually

Instantly get a fraud prediction

View model confidence

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

🧮 Model Insights

Algorithm: LightGBM (fast, scalable gradient boosting)

Handling Imbalance: SMOTE oversampling

Scaler: StandardScaler

Metric Focus: Recall (to reduce false negatives)

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------

-------------------*********-----------------------------------------------


