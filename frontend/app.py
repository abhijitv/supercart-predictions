
import io
import os
import requests
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Predictor")
st.markdown("Enter product and store details below to forecast sales revenue.")

st.subheader("Product Details")
col1, col2 = st.columns(2)

with col1:
    product_weight   = st.number_input("Product Weight (kg)", min_value=0.0, value=12.66, step=0.01)
    product_mrp      = st.number_input("Product MRP (₹)", min_value=0.0, value=117.08, step=0.01)
    product_sugar    = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])

with col2:
    product_area     = st.number_input("Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
    product_id_char  = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
    product_type_cat = st.selectbox("Product Type Category", ["Non Perishables", "Perishables"])

st.subheader("Store Details")
col3, col4 = st.columns(2)

with col3:
    store_size      = st.selectbox("Store Size", ["Medium", "High", "Small"])
    store_type      = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Grocery Store", "Departmental Store"])

with col4:
    store_city_type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    store_age       = st.number_input("Store Age (Years)", min_value=1, max_value=100, value=16)

st.divider()

if st.button("Predict Sales", type="primary", use_container_width=True):
    payload = {
        "Product_Weight":           product_weight,
        "Product_Sugar_Content":    product_sugar,
        "Product_Allocated_Area":   product_area,
        "Product_MRP":              product_mrp,
        "Store_Size":               store_size,
        "Store_Location_City_Type": store_city_type,
        "Store_Type":               store_type,
        "Product_Id_char":          product_id_char,
        "Store_Age_Years":          store_age,
        "Product_Type_Category":    product_type_cat,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()["predicted_sales"]
        st.success(f"### Predicted Sales Revenue: ₹ {prediction:,.2f}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Make sure the Flask API is running.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.subheader("Batch Prediction")
st.markdown("Upload a CSV file with the required feature columns to predict sales for multiple records.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.write("Preview:", df_preview.head())

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        try:
            uploaded_file.seek(0)
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": ("batch.csv", uploaded_file.read(), "text/csv")},
                timeout=30,
            )
            response.raise_for_status()
            predictions = response.json()
            result_df = df_preview.copy()
            result_df["Predicted_Sales"] = [predictions[str(i)] for i in range(len(result_df))]
            st.success("Batch prediction complete!")
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Make sure the Flask API is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
