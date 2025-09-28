import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib

# ---------------------------
# Sidebar - upload file
# ---------------------------
st.sidebar.title("Cab Surge Prediction App")
uploaded_file = st.sidebar.file_uploader("Upload test.csv", type=["csv"])

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.pkl")  # pastikan file model ada

model = load_model()

# ---------------------------
# Main app
# ---------------------------
st.title("🚕 Surge Pricing Prediction with XGBoost")
st.write("Aplikasi ini memprediksi **Surge Pricing Type** berdasarkan fitur customer & trip.")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df_test.head())

    # ---------------------------
    # Preprocessing
    # ---------------------------
    # Hapus kolom ID jika ada
    if "Trip_ID" in df_test.columns:
        df_test = df_test.drop(columns=["Trip_ID"])

    # ---------------------------
    # Prediksi
    # ---------------------------
    preds = model.predict(df_test)
    df_test["Predicted_Surge_Pricing_Type"] = preds + 1  # dari 0-2 jadi 1-3

    st.subheader("Prediction Results")
    st.write(df_test[["Predicted_Surge_Pricing_Type"]].value_counts())

    # Tombol download hasil prediksi
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
    # Pastikan jumlah kolom sama seperti saat training
    feature_names = [
        "Trip_Distance", "Type_of_Cab", "Customer_Since_Months",
        "Life_Style_Index", "Confidence_Life_Style_Index", "Destination_Type",
        "Customer_Rating", "Cancellation_Last_1Month", "Var1", "Var2", "Var3", "Gender"
    ]
    feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    st.bar_chart(feature_importance)

else:
    st.info("Upload test.csv terlebih dahulu untuk melihat prediksi.")
