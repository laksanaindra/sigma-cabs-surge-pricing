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
    # Fitur yang dipakai model saat training
    features = [
        "Trip_Distance", "Type_of_Cab", "Customer_Since_Months",
        "Life_Style_Index", "Confidence_Life_Style_Index", "Destination_Type",
        "Customer_Rating", "Cancellation_Last_1Month", "Var1", "Var2", "Var3", "Gender"
    ]
    return model, encoders, features

model, encoders, features = load_model()

# ---------------------------
# Main app
# ---------------------------
st.title("🚕 Surge Pricing Prediction with XGBoost")
st.write("Prediksi **Surge Pricing Type** berdasarkan fitur customer & trip.")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df_test.head())

    # ---------------------------
    # Preprocessing
    # ---------------------------
    # Hapus kolom ID kalau ada
    if "Trip_ID" in df_test.columns:
        df_test = df_test.drop(columns=["Trip_ID"])

    # Ambil hanya kolom fitur yang dipakai model
    df_test = df_test[features]

    # Kolom kategorikal yang perlu di-encode
    cat_cols = ["Type_of_Cab", "Confidence_Life_Style_Index", "Destination_Type", "Gender"]
    for col in cat_cols:
        if col in df_test.columns:
            le = encoders[col]
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
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    st.bar_chart(feature_importance)

else:
    st.info("Upload test.csv terlebih dahulu untuk melihat prediksi.")
