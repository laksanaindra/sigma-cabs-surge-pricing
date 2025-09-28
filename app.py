import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Sidebar - upload file
# ---------------------------
st.sidebar.title("Cab Surge Prediction App")
uploaded_file = st.sidebar.file_uploader("Upload test.csv", type=["csv"])

# ---------------------------
# Load trained model & encoders
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_xgb_model.pkl")
    encoders = {
        "Type_of_Cab": joblib.load("le_type_of_cab.pkl"),
        "Confidence_Life_Style_Index": joblib.load("le_confidence.pkl"),
        "Destination_Type": joblib.load("le_destination.pkl"),
        "Gender": joblib.load("le_gender.pkl")
    }
    return model, encoders

model, encoders = load_model()

# ---------------------------
# Main app
# ---------------------------
st.title("🚕 Surge Pricing Prediction with XGBoost")
st.write("Prediksi **Surge Pricing Type** berdasarkan fitur customer & trip.")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df_test.head())

    # Hapus kolom ID
    if "Trip_ID" in df_test.columns:
        df_test = df_test.drop(columns=["Trip_ID"])

    # Apply saved LabelEncoders
    for col, le in encoders.items():
        if col in df_test.columns:
            df_test[col] = le.transform(df_test[col].astype(str))

    # ---------------------------
    # Prediksi
    # ---------------------------
    preds = model.predict(df_test)
    df_test["Predicted_Surge_Pricing_Type"] = preds + 1  # dari 0-2 jadi 1-3

    st.subheader("Prediction Results")
    st.write(df_test[["Predicted_Surge_Pricing_Type"]].value_counts())

    # ---------------------------
    # Download hasil prediksi
    # ---------------------------
    st.download_button(
        label="Download Predictions",
        data=df_test.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )

    # ---------------------------
    # Feature Importance (XGBoost built-in)
    # ---------------------------
    st.subheader("Feature Importance")
    feature_importance = pd.Series(model.feature_importances_, index=df_test.columns).sort_values(ascending=True)
    
    st.bar_chart(feature_importance)

else:
    st.info("Upload test.csv terlebih dahulu untuk melihat prediksi.")
