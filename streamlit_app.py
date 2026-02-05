import os
from typing import Any, Dict

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("IRIS_API_URL", "http://localhost:8000")


def call_api(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = api_url.rstrip("/") + "/predict"
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Iris Predictor", page_icon="🌿", layout="centered")
st.title("Iris Species Predictor")
st.write("Entre les 4 mesures pour predire l'espece.")

with st.sidebar:
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    st.caption("Ex: http://localhost:8000")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("sepal_length", min_value=0.0, value=5.1, step=0.1)
    petal_length = st.number_input("petal_length", min_value=0.0, value=1.4, step=0.1)
with col2:
    sepal_width = st.number_input("sepal_width", min_value=0.0, value=3.5, step=0.1)
    petal_width = st.number_input("petal_width", min_value=0.0, value=0.2, step=0.1)

if st.button("Predict", type="primary"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        data = call_api(api_url, payload)
    except requests.RequestException as exc:
        st.error(f"Erreur API: {exc}")
    else:
        st.success(f"Prediction: {data.get('species_pred', 'N/A')}")
        decision = data.get("decision")
        if decision:
            st.write(decision)
        confidence = data.get("confidence")
        if confidence is not None:
            st.write(f"Confiance: {confidence:.4f}")

        probabilities = data.get("probabilities")
        if probabilities:
            rows = [
                {"classe": k, "probabilite": float(v)} for k, v in probabilities.items()
            ]
            rows = sorted(rows, key=lambda r: r["probabilite"], reverse=True)
            st.subheader("Probabilites")
            st.table(rows)

        with st.expander("Reponse brute"):
            st.json(data)
