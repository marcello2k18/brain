import os
import sys

# 1. SOLUSI REKURSI & LEGACY
sys.setrecursionlimit(10000)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="centered")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_all_models():
    try:
        # Bersihkan sesi TF
        tf.keras.backend.clear_session()
        
        # Menggunakan tf.keras.models agar lebih sinkron dengan versi 2.15.0
        m_res = tf.keras.models.load_model('models/resnet_extractor.h5', compile=False)
        m_dense = tf.keras.models.load_model('models/densenet_extractor.h5', compile=False)
        
        # Load file-file preprocessing (.pkl)
        with open('models/scaler_resnet.pkl', 'rb') as f:
            s_res = pickle.load(f)
        with open('models/scaler_dense.pkl', 'rb') as f:
            s_dense = pickle.load(f)
        with open('models/pca_fusion.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('models/selector_fusion.pkl', 'rb') as f:
            sel = pickle.load(f)
        with open('models/svm_fusion.pkl', 'rb') as f:
            svm = pickle.load(f)
        
        return m_res, m_dense, s_res, s_dense, pca, sel, svm
    except Exception as e:
        # Menampilkan detail error di UI jika gagal
        st.error(f"⚠️ Detail Error Saat Memuat Model: {e}")
        return None

# Load Models
models = load_all_models()

if models:
    m_res, m_dense, s_res, s_dense, pca_f, select_f, svm_f = models

    st.title("🧠 Klasifikasi Tumor Otak MRI")
    st.markdown("Dashboard Skripsi: **Feature Fusion & SVM**")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Pilih gambar MRI...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar Input', use_container_width=True)
        
        if st.button('Mulai Analisis Sekarang', use_container_width=True):
            with st.spinner('Sedang memproses...'):
                try:
                    # 1. Preprocessing
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0).astype('float32')
                    
                    # 2. Pipeline Ekstraksi Fitur
                    f_res = m_res.predict(img_batch, verbose=0)
                    f_dense = m_dense.predict(img_batch, verbose=0)
                    
                    # 3. Scaling & Fusion
                    f_res_s = s_res.transform(f_res)
                    f_dense_s = s_dense.transform(f_dense)
                    combined = np.hstack([f_res_s, f_dense_s])
                    
                    # 4. Reduksi & Prediksi Akhir
                    f_pca = pca_f.transform(combined)
                    f_final = select_f.transform(f_pca)
                    pred = svm_f.predict(f_final)
                    
                    # Hasil Prediksi
                    st.markdown("---")
                    st.subheader("Hasil Diagnosis:")
                    hasil = str(pred[0])
                    if hasil.lower() == 'no tumor':
                        st.balloons()
                        st.success(f"### **{hasil}**")
                    else:
                        st.error(f"### **Terdeteksi: {hasil}**")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
else:
    st.warning("Gagal memuat sistem. Coba jalankan: `pip install keras==2.15.0` di terminal.")