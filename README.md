Transaction Fraud Detection Model 💳

This project delivers a machine learning model to detect fraudulent financial transactions, with a specific focus on the expansion of Blocker Fraud Company in Brazil. The solution includes data analysis, feature engineering, model training, and a deployed API accessible via a Streamlit application.

🚀 Key Features

Data Analysis: In-depth exploratory data analysis (EDA) to understand transaction patterns and identify key features.
Feature Engineering: Creation of new variables to capture valuable information, such as balance changes before and after transactions.
Model Training: A robust XGBoost model was trained and fine-tuned to achieve high performance in detecting fraud.
Business-Oriented Metrics: The model's performance is evaluated not only on standard metrics (Precision, Accuracy) but also on business-specific metrics like expected profit and loss.
Deployment Ready: The model and its pre-processing pipeline are saved for easy integration into an API, with a live demo available on Streamlit.

📈 Model Performance

The final XGBoost model demonstrated strong performance on unseen data, as detailed below:
Balanced Accuracy: 0.85
Precision: 0.84
Recall: 0.85
F1 Score: 0.84
Kappa Score: 0.84

💰 Business Impact

Based on a test dataset, the model shows a significant improvement in profitability for the Blocker Fraud Company compared to its current method.
Metric	Using the Model	Current Method (isFlaggedFraud)
Expected Revenue	R$ 1,778,591.70	R$ 0.00
Expected Loss	R$ 32,836.56	R$ 33,680,683.00
Expected Profit	R$ 1,745,755.14	R$ -33,680,683.00
Note: All monetary values are based on the test data used for evaluation and are represented in Brazilian Reais (R$).

🛠️ Project Structure

The project code is organized into the following sections:
Business Understanding: Defines the project's goals, the problem, and the key business questions to be answered.
Data Description & Pre-processing: Details the dataset, cleans the data, and performs initial data type conversions.
Feature Engineering: Creates new features to enhance model performance.
Exploratory Data Analysis (EDA): Visualizes and analyzes the data to validate hypotheses.
Data Preparation: Splits the data and applies scaling and encoding transformations.
Feature Selection: Uses a robust method to select the most relevant features for the model.
Machine Learning Modeling: Trains and evaluates several classification models, comparing their performance.
Hyperparameter Fine-Tuning: Optimizes the best-performing model (XGBoost) to maximize performance.
Conclusions & Business Insights: Presents the final model's performance and answers the initial business questions.
Model Deployment: Provides the code and necessary files to deploy the model for inference via an API.

💻 How to Run the Project

Prerequisites
You need to have Python installed. We recommend using a virtual environment.

Installation
Clone this repository:

Bash

git clone https://github.com/anjali5Xcode/Fraud_Transaction_Detection.git
cd Fraud_Transaction_Detection
Install the required packages:

Bash

pip install -r requirements.txt
(Note: You'll need to create a requirements.txt file from the imported libraries in the notebook, like pandas, numpy, scikit-learn, xgboost, lightgbm, streamlit, etc.)

Execution
The core logic of the project is contained within the transaction_fraud_detection_cycle1 (1).py notebook. You can run it locally in a Jupyter environment.

To run the Streamlit application for a live demo:
Ensure you have the model and scaler files (fraud_model.pkl, minmaxscaler_cycle1.joblib, onehotencoder_cycle1.joblib) saved in the same directory.
Run the Streamlit application from your terminal:

Bash

streamlit run main.py
The application will open in your web browser, allowing you to input transaction details and get real-time fraud predictions.

🔗 Live Demo
You can access a live version of the application here:
https://fraudtransactiondetection-fa9erdbmtf8uwgtmh5u6sc.streamlit.app/
