import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask(__name__)

model = joblib.load("/content/drive/MyDrive/models/backend_files/superkart_model.joblib")

FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@superkart_api.get("/")
def health():
    return jsonify({"status": "ok", "model": type(model[-1]).__name__})


@superkart_api.post("/v1/predict")
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data], columns=FEATURES)
    prediction = model.predict(df)[0]
    return jsonify({"predicted_sales": round(float(prediction), 2)})


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file provided. Send CSV as 'file' field."}), 400
    df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400
    predictions = model.predict(df[FEATURES])
    result = {str(i): round(float(p), 2) for i, p in enumerate(predictions)}
    return jsonify(result)


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
