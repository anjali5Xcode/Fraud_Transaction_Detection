💳 Transaction Fraud Detection using Machine Learning

📌 Project Overview

Financial fraud is a serious problem in digital banking and online transactions.
This project builds a machine learning pipeline to detect fraudulent transactions based on transaction amount, balances, and transaction type.

The project includes:

📊 Exploratory Data Analysis (EDA)

🛠 Feature Engineering & Preprocessing

🤖 Model Training & Evaluation (Random Forest, XGBoost, etc.)

📈 Performance Metrics & Visualization (Confusion Matrix, ROC, Precision-Recall, SHAP Explainability)

🌐 Deployment-ready App built with Streamlit for interactive predictions


🛠️ Tech Stack

Python 3.9+

Pandas, NumPy → Data manipulation

Matplotlib, Seaborn → Visualization

Scikit-learn → ML algorithms & preprocessing

XGBoost → Gradient boosting model

Joblib → Model serialization

Streamlit → Interactive web app

SHAP → Model explainability

📂 Project Structure

fraud_detection_app/
│── app.py                        # Streamlit app
│── fraud.py                      # Fraud pipeline class
│── transaction_fraud_detection_cycle1.py   # Full training script
│── fraud_model.pkl                # Trained model
│── minmaxscaler_cycle1.joblib     # Scaler object
│── onehotencoder_cycle1.joblib    # Encoder object
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation


🚀 How to Run

1️⃣ Clone Repository
git clone https://github.com/anjali5Xcode/fraud-detection-app.git
cd fraud-detection-app

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py


App will be available at 👉 http://localhost:8501

📊 Model Training (Notebook / Script)


The training pipeline includes:

Data Cleaning → Fixing column names, handling missing values

Feature Engineering → Creating new balance difference features

Preprocessing → Scaling numeric values, encoding categorical variables

Modeling → Random Forest, XGBoost with hyperparameter tuning

Evaluation → F1-score, ROC-AUC, Precision-Recall Curve

Explainability → SHAP values for feature importance


🌐 Streamlit App (Demo)

The app allows you to:

Input transaction details (amount, balances, transaction type)

Predict whether the transaction is Fraudulent 🚨 or Legitimate ✅

Explore processed data in an expandable section


📈 Results

Best model: XGBoost

Achieved high ROC-AUC & F1 score on imbalanced dataset

SHAP analysis highlighted transaction type & balance differences as key fraud indicators


🔮 Future Improvements

Add real-time fraud detection API (FastAPI/Flask)

Deploy app on Streamlit Cloud / Heroku / AWS

Integrate with transaction logging system


👩‍💻 Author

Developed by Anjali
