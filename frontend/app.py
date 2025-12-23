import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Mantenimiento predictivo ACIF104", layout="wide")

st.title("Dashboard de mantenimiento predictivo")
st.markdown(
    "Este frontend consume la API FastAPI del backend para obtener predicciones "
    "de probabilidad de falla y métricas globales del modelo."
)

st.sidebar.header("Parámetros del equipo")
air_temp = st.sidebar.slider("Air temperature [K]", 290.0, 320.0, 300.0, 0.1)
proc_temp = st.sidebar.slider("Process temperature [K]", 300.0, 330.0, 310.0, 0.1)
speed = st.sidebar.slider("Rotational speed [rpm]", 1200, 2000, 1500, 10)
torque = st.sidebar.slider("Torque [Nm]", 20.0, 60.0, 40.0, 0.1)
wear = st.sidebar.slider("Tool wear [min]", 0, 300, 150, 5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicción de falla")
    if st.button("Calcular probabilidad de falla"):
        payload = {
            "air_temperature_k": air_temp,
            "process_temperature_k": proc_temp,
            "rotational_speed_rpm": speed,
            "torque_nm": torque,
            "tool_wear_min": wear,
        }
        try:
            resp = requests.post(f"{API_URL}/predict_falla", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                prob = data["failure_probability"]
                st.metric("Probabilidad de falla", f"{prob*100:.2f} %")
                if prob >= 0.5:
                    st.error("Alerta: equipo en alto riesgo de falla, revisar planificación de mantenimiento.")
                else:
                    st.success("Riesgo controlado. Continuar monitoreo periódico.")
            else:
                st.error(f"Error al consultar la API: {resp.text}")
        except Exception as e:
            st.error(f"No se pudo conectar con la API. ¿Está ejecutándose el backend? ({e})")

with col2:
    st.subheader("Métricas globales del modelo")
    if st.button("Actualizar métricas"):
        try:
            resp = requests.get(f"{API_URL}/metrics")
            data = resp.json()
            if "detail" in data:
                st.warning(data["detail"])
            else:
                st.json(data)
        except Exception as e:
            st.error(f"No se pudo conectar con la API. ¿Está ejecutándose el backend? ({e})")

st.markdown("---")
st.markdown(
    "Para más detalle sobre la explicación del modelo (SHAP), consulte los notebooks y figuras "
    "disponibles en la carpeta `notebooks/` (por ejemplo, `notebooks/figuras/`)."
)
