import streamlit as st
import pandas as pd
import joblib
import inflection

class Fraud:

    def __init__(self):
        # Load fitted scaler & encoder
        self.minmaxscaler = joblib.load("minmaxscaler_cycle1.joblib")
        self.onehotencoder = joblib.load("onehotencoder_cycle1.joblib")

        # Numeric columns
        self.num_columns = [
            "amount", "oldbalance_org", "newbalance_orig",
            "oldbalance_dest", "newbalance_dest",
            "diff_new_old_balance", "diff_new_old_destiny"
        ]

        # Final columns used in training
        self.final_columns_selected = [
            "step", "oldbalance_org", "newbalance_orig",
            "newbalance_dest", "diff_new_old_balance",
            "diff_new_old_destiny",
            "type_TRANSFER", "type_CASH_OUT",
            "type_PAYMENT", "type_DEBIT", "type_CASH_IN"
        ]

    def data_cleaning(self, df):
        """Convert column names to snake_case (match training)."""
        cols_old = df.columns.tolist()
        snakecase = lambda i: inflection.underscore(i)
        cols_new = list(map(snakecase, cols_old))
        df.columns = cols_new
        return df

    def feature_engineering(self, df):
        """Add engineered features."""
        df["diff_new_old_balance"] = df["newbalance_orig"] - df["oldbalance_org"]
        df["diff_new_old_destiny"] = df["newbalance_dest"] - df["oldbalance_dest"]
        return df.drop(columns=["name_orig", "name_dest"], errors="ignore")

    def data_preparation(self, df):
        """Apply scaling + category_encoders OneHotEncoder."""
        df[self.num_columns] = self.minmaxscaler.transform(df[self.num_columns])
        df_encoded = self.onehotencoder.transform(df)

        # Ensure all training columns exist
        for col in self.final_columns_selected:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        return df_encoded[self.final_columns_selected]

    def get_prediction(self, model, original_data, test_data):
        """Return predictions with original input."""
        pred = model.predict(test_data)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_data)[:, 1]
        output = original_data.copy()
        output["prediction"] = pred
        if proba is not None:
            output["fraud_probability"] = proba
        return output


# -------------------------------
# Load trained model & pipeline
# -------------------------------
model = joblib.load("fraud_model.pkl")
pipeline = Fraud()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💳 Transaction Fraud Detection")

st.sidebar.header("Enter Transaction Details")

# Sidebar Inputs
step = st.sidebar.number_input("Step (time unit of transaction)", min_value=1, value=1, step=1)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
oldbalance_org = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=500.0, step=0.01)
newbalance_orig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=400.0, step=0.01)
oldbalance_dest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=200.0, step=0.01)
newbalance_dest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=300.0, step=0.01)
transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT", "CASH_IN"]
)

# Mock IDs (required but not important for prediction)
name_orig = "C123456"
name_dest = "M123456"

# Prediction Button
if st.sidebar.button("Predict Fraud"):
    try:
        # Create DataFrame similar to training data
        input_data = pd.DataFrame([{
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
            "nameOrig": name_orig,
            "nameDest": name_dest
        }])

        # Apply Fraud pipeline
        df1 = pipeline.data_cleaning(input_data)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        prediction_df = pipeline.get_prediction(model, input_data, df3)

        # Get prediction result
        pred = prediction_df["prediction"].iloc[0]
        result = "🚨 Fraudulent Transaction" if pred == 1 else "✅ Legitimate Transaction"

        # Show result
        st.subheader("Prediction Result:")
        if pred == 1:
            st.error(result)
        else:
            st.success(result)

        # Show fraud probability if available
        if "fraud_probability" in prediction_df.columns:
            prob = prediction_df["fraud_probability"].iloc[0]
            st.write(f"Fraud Probability: **{prob:.2%}**")

        # Show processed input
        with st.expander("🔍 Processed Input Data"):
            st.write(prediction_df)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
