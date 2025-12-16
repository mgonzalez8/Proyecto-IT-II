import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------
# Configuración
# ----------------------------
st.set_page_config(page_title="Diagnóstico Caña de Azúcar", page_icon="🌿", layout="centered")

MODEL_PATH = "models/best_model_CNN_Simple.h5"
IMG_SIZE = (224, 224)

# IMPORTANTE:
# Keras flow_from_directory asigna índices en orden alfabético de carpetas.
# Para: Healthy, Mosaic, RedRot, Rust, Yellow -> 0..4 (normalmente)
CLASSES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]


@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0  # rescale=1./255
    arr = np.expand_dims(arr, axis=0)               # (1,224,224,3)
    return arr


def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]          # (5,)
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


# ----------------------------
# UI
# ----------------------------
st.title(" Detección de enfermedades en hojas de caña de azúcar")
st.write("Sube una imagen (JPG/PNG) y el modelo **CNN Simple** predecirá la clase.")

# Mostrar ruta del modelo (editable si lo deseas)
model_path = st.text_input("Ruta del modelo (.h5)", value=MODEL_PATH)

uploaded = st.file_uploader("📷 Cargar imagen de hoja", type=["jpg", "jpeg", "png"])
run_btn = st.button("🔎 Predecir", use_container_width=True)

if run_btn:
    # Validaciones
    if not model_path or not os.path.exists(model_path):
        st.error(f"No se encontró el modelo en: {model_path}")
        st.stop()

    if uploaded is None:
        st.warning("Primero sube una imagen.")
        st.stop()

    # Cargar modelo
    try:
        model = load_model_cached(model_path)
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        st.stop()

    # Leer imagen
    try:
        img = Image.open(uploaded)
    except Exception as e:
        st.error(f"No pude abrir la imagen: {e}")
        st.stop()

    st.image(img, caption="Imagen cargada", use_container_width=True)

    # Predicción
    pred_class, conf, probs = predict(model, img)

    st.subheader("✅ Resultado")
    st.metric("Clase predicha", pred_class, f"{conf*100:.2f}% confianza")

    st.subheader("📊 Probabilidades por clase")
    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    st.bar_chart(prob_dict)

    st.caption("Clases: Healthy, Mosaic, RedRot, Rust, Yellow")
