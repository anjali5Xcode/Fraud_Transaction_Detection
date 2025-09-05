import streamlit as st
import pandas as pd
import joblib
import inflection

# -------------------------------
# Fraud Class Definition
# -------------------------------
class Fraud:

    def __init__(self):
        # Load pre-trained scalers/encoders
        self.minmaxscaler = joblib.load("minmaxscaler_cycle1.joblib")
        self.onehotencoder = joblib.load("onehotencoder_cycle1.joblib")

        # Define the numeric columns used during training
        self.num_columns = [
            "amount", "oldbalance_org", "newbalance_orig",
            "oldbalance_dest", "newbalance_dest",
            "diff_new_old_balance", "diff_new_old_destiny"
        ]

        # Columns selected in final training
        self.final_columns_selected = [
            "step", "oldbalance_org", "newbalance_orig",
            "newbalance_dest", "diff_new_old_balance",
            "diff_new_old_destiny", "type_TRANSFER"
        ]

    def data_cleaning(self, df):
        """Convert column names to snake_case."""
        cols_old = df.columns.tolist()
        snakecase = lambda i: inflection.underscore(i)
        cols_new = list(map(snakecase, cols_old))
        df.columns = cols_new
        return df

    def feature_engineering(self, df):
        """Add derived features like balance differences."""
        df["step_days"] = df["step"] / 24
        df["step_weeks"] = df["step"] / (24 * 7)

        df["diff_new_old_balance"] = df["newbalance_orig"] - df["oldbalance_org"]
        df["diff_new_old_destiny"] = df["newbalance_dest"] - df["oldbalance_dest"]

        df["name_orig"] = df["name_orig"].apply(lambda i: i[0])
        df["name_dest"] = df["name_dest"].apply(lambda i: i[0])

        # Drop unnecessary columns
        return df.drop(columns=["name_orig", "name_dest", "step_weeks", "step_days"], axis=1)

    def data_preparation(self, df):
        """Apply scaling + encoding in the same way as training."""
        # Scale numeric columns
        df[self.num_columns] = self.minmaxscaler.transform(df[self.num_columns])

        # Apply one-hot encoding
        df_encoded = pd.DataFrame(
            self.onehotencoder.transform(df).toarray(),
            columns=self.onehotencoder.get_feature_names_out(df.columns)
        )

        # Align with training columns - ensure all selected columns are present
        for col in self.final_columns_selected:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # add missing columns

        return df_encoded[self.final_columns_selected]

    def get_prediction(self, model, original_data, test_data):
        """Generate predictions and return DataFrame with results."""
        pred = model.predict(test_data)
        original_data = original_data.copy()
        original_data["prediction"] = pred
        return original_data


# -------------------------------
# Streamlit App
# -------------------------------

# Load trained model
model = joblib.load("fraud_model.pkl")
pipeline = Fraud()

# App Title
st.title("💳 Transaction Fraud Detection")

# Sidebar Inputs
st.sidebar.header("Enter Transaction Details")
step = st.sidebar.number_input("Step (time unit of transaction)", min_value=1, step=1)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.01)
oldbalance_org = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, step=0.01)
newbalance_orig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, step=0.01)
oldbalance_dest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, step=0.01)
newbalance_dest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, step=0.01)
transaction_type = st.sidebar.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT", "CASH_IN"])

# Mock IDs (required by pipeline, but not important for prediction)
name_orig = "C123456"
name_dest = "M123456"

# Prediction Button
if st.sidebar.button("Predict Fraud"):
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
    st.success(result)

    # Optional: show processed input
    with st.expander("🔍 Processed Input Data"):
        st.write(prediction_df)
