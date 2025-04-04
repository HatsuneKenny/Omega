import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml

from sklearn.ensemble import RandomForestClassifier

# ⚙️ Načtení konfigurace
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data_path"]
MODEL_PATH = config["model_path"]
ENCODING_PATHS = config["encoding_paths"]
DEFAULT_GRID = config["default_values"]["grid_position"]

# 📂 Načtení modelu a encoderů
model = joblib.load(MODEL_PATH)
encoder_circuit = joblib.load(ENCODING_PATHS["circuit"])
encoder_constructor = joblib.load(ENCODING_PATHS["constructor"])
encoder_driver = joblib.load(ENCODING_PATHS["driver"])

# 📃 Dataset pro náhled možností
df = pd.read_csv(DATA_PATH)

available_seasons = sorted(df["Season"].unique())
circuit_names = encoder_circuit.classes_
constructor_names = encoder_constructor.classes_

# 🌟 Stylová hlavička
st.set_page_config(page_title="F1 Winner Predictor", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>🏎️ F1 Winner Predictor</h1>
    <p style='text-align: center; color: gray;'>Zjisti pravděpodobnost výhry jezdce podle historických dat</p>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 Zadej parametry závodu")

col1, col2 = st.columns(2)
with col1:
    season = st.selectbox("Sezóna", available_seasons)
    constructor = st.selectbox("Tým", constructor_names)
with col2:
    circuit = st.selectbox("Okruh", circuit_names)
    grid_pos = st.number_input("Startovní pozice", min_value=1, max_value=30, value=DEFAULT_GRID)

# Filtrace jezdců podle vstupu z nešifrovaného datasetu
df_raw = pd.read_csv(DATA_PATH)
filtered = df_raw[
    (df_raw["Season"] == season) &
    (df_raw["Constructor"] == constructor) &
    (df_raw["Circuit"] == circuit)
]
driver_names = sorted(filtered["Driver"].unique())

if len(driver_names) == 0:
    st.error("⚠️ Pro tuto kombinaci sezóny, týmu a okruhu nejsou dostupní žádní jezdci.")
    st.stop()

driver = st.selectbox("Jezdec", driver_names)

st.markdown("---")

if st.button("🔬 Spustit predikci"):
    # Kontrola kombinace v datech
    row_exists = df_raw[
        (df_raw["Season"] == season) &
        (df_raw["Circuit"] == circuit) &
        (df_raw["Constructor"] == constructor) &
        (df_raw["Driver"] == driver)
    ].shape[0] > 0

    if not row_exists:
        st.error("⚠️ Tato kombinace sezóny, okruhu, týmu a jezdce neexistuje v datech.")
    else:
        # 📊 Kódování vstupů
        circuit_encoded = encoder_circuit.transform([circuit])[0]
        constructor_encoded = encoder_constructor.transform([constructor])[0]
        driver_encoded = encoder_driver.transform([driver])[0]

        input_data = np.array([[season, circuit_encoded, grid_pos, constructor_encoded, driver_encoded]])
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.metric(label=f"🚗 Šance na výhru pro {driver}", value=f"{probability * 100:.2f}%")
        st.success(f"Predikce proběhla úspěšně pro závod v **{circuit}**, sezóna **{season}**.")