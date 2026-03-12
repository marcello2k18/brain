import os
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        model_path = 'resnet_model.onnx'
        if not os.path.exists(model_path):
            st.error(f"❌ File {model_path} tidak ditemukan")
            return None
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

session = load_model()

if session:
    st.title("🧠 Klasifikasi Tumor Otak MRI")
    st.markdown("Dashboard Skripsi: **ResNet50**")
    st.markdown("---")

    uploaded_file = st.file_uploader("Pilih gambar MRI...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar Input', use_container_width=True)

        if st.button('Mulai Analisis Sekarang', use_container_width=True):
            with st.spinner('Sedang memproses...'):
                try:
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0).astype('float32')

                    input_name = session.get_inputs()[0].name
                    output = session.run(None, {input_name: img_batch})[0]

                    pred_idx = int(np.argmax(output[0]))
                    confidence = float(np.max(output[0])) * 100
                    label = CLASS_NAMES.get(pred_idx, f"Kelas {pred_idx}")

                    st.markdown("---")
                    st.subheader("Hasil Diagnosis:")

                    if "no tumor" in label.lower():
                        st.balloons()
                        st.success(f"### ✅ **{label}**")
                    else:
                        st.error(f"### ⚠️ **Terdeteksi: {label}**")

                    st.metric("Confidence", f"{confidence:.2f}%")

                    with st.expander("📊 Detail Probabilitas"):
                        for i, prob in enumerate(output[0]):
                            name = CLASS_NAMES.get(i, f"Kelas {i}")
                            st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
else:
    st.error("❌ Gagal memuat model.")
