import os
import io
import socket
import cv2
import joblib
import qrcode
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Penataan Sepatu (Extra Trees & Camo Studio)",
    page_icon="👟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 26px;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        font-size: 14px;
        color: #64748B;
        margin-bottom: 18px;
    }
    .model-badge {
        background-color: #EEF2F6;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 13px;
        text-align: center;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 15px;
    }
    .card-rapi {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
        margin: 15px 0;
    }
    .card-berantakan {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.4);
        margin: 15px 0;
    }
    .status-text {
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .prob-text {
        font-size: 16px;
        font-weight: 500;
        opacity: 0.95;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNGSI MEMUAT MODEL EXTRA TREES CLASSIFIER
# ==============================================================================
@st.cache_resource
def load_extratrees_bundle():
    model_path = 'model_extratrees_penataan_sepatu.pkl'
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan!")
        return None
    return joblib.load(model_path)

# ==============================================================================
# EKSTRAKSI 48 FITUR MULTIMODAL LENGKAP
# ==============================================================================
def extract_48_features(img_bgr, kmeans_bovw, fitur_cols):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_res = cv2.resize(img_gray, (256, 256))
    
    # 1. 9 Fitur RGB
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    fitur_dict = {
        'mean_r': float(np.mean(r)), 'mean_g': float(np.mean(g)), 'mean_b': float(np.mean(b)),
        'std_r': float(np.std(r)), 'std_g': float(np.std(g)), 'std_b': float(np.std(b)),
        'skew_r': float(skew(r.ravel())), 'skew_g': float(skew(g.ravel())), 'skew_b': float(skew(b.ravel()))
    }
    
    # 2. 5 Fitur Tekstur GLCM CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_res)
    glcm = graycomatrix(img_clahe, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    fitur_dict['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
    p_norm = glcm / (np.sum(glcm) + 1e-15)
    fitur_dict['glcm_entropy'] = float(-np.sum(p_norm * np.log2(p_norm + 1e-15)))
    fitur_dict['glcm_correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
    fitur_dict['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
    fitur_dict['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
    
    # 3. 4 Fitur Edge
    canny = cv2.Canny(img_clahe, 50, 150)
    fitur_dict['edge_density'] = float(np.mean(canny > 0))
    sobelx = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_clahe, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    fitur_dict['sobel_mean'] = float(np.mean(sobel_mag))
    fitur_dict['sobel_std'] = float(np.std(sobel_mag))
    laplacian = cv2.Laplacian(img_clahe, cv2.CV_64F)
    fitur_dict['laplacian_var'] = float(np.var(laplacian))
    
    # 4. 6 Fitur Shape Kontur
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fitur_dict['num_contours'] = len(contours)
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
    fitur_dict['contour_area_mean'] = float(np.mean(areas)) if len(areas) > 0 else 0.0
    fitur_dict['contour_area_std'] = float(np.std(areas)) if len(areas) > 0 else 0.0
    solidities = [cv2.contourArea(c)/cv2.contourArea(cv2.convexHull(c)) for c in contours if cv2.contourArea(c) > 20 and cv2.contourArea(cv2.convexHull(c)) > 0]
    fitur_dict['contour_solidity'] = float(np.mean(solidities)) if len(solidities) > 0 else 0.0
    hu = cv2.HuMoments(cv2.moments(canny)).flatten()
    fitur_dict['hu_moment_1'] = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-15))
    fitur_dict['hu_moment_2'] = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-15))
    
    # 5. 10 Fitur HOG
    ang = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)[1] % 180.0
    h_hist, _ = np.histogram(ang, bins=np.linspace(0, 180, 9), weights=sobel_mag)
    h_norm = h_hist / (np.sum(h_hist) + 1e-15)
    fitur_dict['hog_entropy'] = float(-np.sum(h_norm * np.log2(h_norm + 1e-15)))
    fitur_dict['hog_ortho_ratio'] = float((h_norm[0] + h_norm[3] + h_norm[7]) / (np.sum(h_norm) + 1e-15))
    for b in range(8):
        fitur_dict[f'hog_bin_{b}'] = float(h_norm[b])
        
    # 6. 14 Fitur SIFT & BoVW
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_res, None)
    if kp is not None and len(kp) > 0:
        resps = [k.response for k in kp]
        sizes = [k.size for k in kp]
        xs = [k.pt[0] for k in kp]
        ys = [k.pt[1] for k in kp]
        fitur_dict['sift_num_keypoints'] = float(len(kp))
        fitur_dict['sift_response_mean'] = float(np.mean(resps))
        fitur_dict['sift_response_std'] = float(np.std(resps))
        fitur_dict['sift_size_mean'] = float(np.mean(sizes))
        fitur_dict['sift_size_std'] = float(np.std(sizes))
        fitur_dict['sift_spatial_spread'] = float(np.std(xs) * np.std(ys))
    else:
        fitur_dict['sift_num_keypoints'] = 0.0
        fitur_dict['sift_response_mean'] = 0.0
        fitur_dict['sift_response_std'] = 0.0
        fitur_dict['sift_size_mean'] = 0.0
        fitur_dict['sift_size_std'] = 0.0
        fitur_dict['sift_spatial_spread'] = 0.0
        
    bovw_hist = np.zeros(8, dtype=np.float32)
    if des is not None and len(des) > 0:
        preds = kmeans_bovw.predict(des)
        for p in preds:
            bovw_hist[p] += 1
        bovw_hist = bovw_hist / (np.sum(bovw_hist) + 1e-15)
    for w in range(8):
        fitur_dict[f'sift_bovw_w{w}'] = float(bovw_hist[w])
        
    x_vec = np.array([[fitur_dict[c] for c in fitur_cols]])
    return x_vec, fitur_dict

# ==============================================================================
# LOGIKA UTAMA
# ==============================================================================
def main():
    st.markdown('<div class="main-title">👟 Klasifikasi Penataan Sepatu</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Integrasi Camo Studio (Kamera HP) & Extra Trees Classifier</div>', unsafe_allow_html=True)
    
    bundle = load_extratrees_bundle()
    if bundle is None:
        st.error("Model Extra Trees belum tersedia!")
        return

    model = bundle['model']
    scaler = bundle['scaler']
    kmeans_bovw = bundle['kmeans_bovw']
    fitur_cols = bundle['fitur_cols']
    optimal_th = bundle.get('optimal_threshold', 0.48)
    acc = bundle.get('accuracy_cv', 0.9233)
    roc = bundle.get('roc_auc', 0.9665)
    
    st.markdown(f'<div class="model-badge">⚡ Metode Aktif: <b>Extra Trees Classifier (150 Pohon)</b> | Akurasi CV: <b>{acc:.2%}</b> | ROC AUC: <b>{roc:.4f}</b></div>', unsafe_allow_html=True)

    with st.expander("📱 Cara Penggunaan dengan Camo Studio di HP", expanded=False):
        st.markdown("""
        **Petunjuk:**
        1. Buka aplikasi **Camo** di HP Anda dan **Camo Studio** di laptop.
        2. Hubungkan HP ke laptop (USB atau Wi-Fi).
        3. Anda bisa memilih kamera Camo langsung di bawah ini atau menjalankan file desktop **`camo_kamera_hp.py`**.
        """)

    tab_camera, tab_upload = st.tabs(["📷 Kamera HP / Camo Studio", "📁 Upload File Foto"])
    
    img_bgr = None
    
    with tab_camera:
        st.write("Arahkan kamera ke penataan sepatu, lalu tekan **Take Photo**.")
        camera_image = st.camera_input("Ambil Foto dari Kamera", label_visibility="collapsed")
        if camera_image is not None:
            bytes_data = camera_image.getvalue()
            img_array = np.frombuffer(bytes_data, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
    with tab_upload:
        uploaded_file = st.file_uploader("Unggah foto penataan sepatu:", type=['jpg', 'jpeg', 'png', 'webp'])
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            img_array = np.frombuffer(bytes_data, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            st.image(bytes_data, caption="Foto Terpilih", use_container_width=True)

    if img_bgr is not None:
        with st.spinner("Mengekstrak 48 fitur multimodal & mengklasifikasi dengan Extra Trees..."):
            x_vec, fitur_dict = extract_48_features(img_bgr, kmeans_bovw, fitur_cols)
            x_scaled = scaler.transform(x_vec)
            
            # Prediksi Probabilitas
            prob_rapi = model.predict_proba(x_scaled)[0, 1]
            prob_berantakan = 1.0 - prob_rapi
            is_rapi = prob_rapi >= optimal_th
            confidence = (prob_rapi if is_rapi else prob_berantakan) * 100
            
        st.markdown("---")
        
        # Tampilan Hasil
        if is_rapi:
            st.markdown(f"""
            <div class="card-rapi">
                <div class="status-text">🟢 PENATAAN RAPI</div>
                <div class="prob-text">Tingkat Keyakinan Extra Trees: <b>{confidence:.1f}%</b></div>
                <div style="font-size: 13px; margin-top: 6px; opacity: 0.9;">
                    Probabilitas Rapi: {prob_rapi:.2%} (Ambang Batas Optimal: {optimal_th:.2f})
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-berantakan">
                <div class="status-text">🔴 PENATAAN BERANTAKAN</div>
                <div class="prob-text">Tingkat Keyakinan Extra Trees: <b>{confidence:.1f}%</b></div>
                <div style="font-size: 13px; margin-top: 6px; opacity: 0.9;">
                    Probabilitas Berantakan: {prob_berantakan:.2%} (Ambang Batas Optimal: {optimal_th:.2f})
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Distribusi Probabilitas
        st.write("📊 **Distribusi Probabilitas Kelas (Extra Trees):**")
        col_r, col_b = st.columns(2)
        with col_r:
            st.metric("Peluang Rapi", f"{prob_rapi:.1%}")
            st.progress(float(prob_rapi))
        with col_b:
            st.metric("Peluang Berantakan", f"{prob_berantakan:.1%}")
            st.progress(float(prob_berantakan))
                
        # Rincian 48 Fitur
        with st.expander("🔍 Nilai Fitur Kunci (GLCM, Edge, HOG, SIFT)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Tekstur GLCM (Slide 19 & 20):**")
                st.write(f"- Kontras: `{fitur_dict['glcm_contrast']:.2f}`")
                st.write(f"- Entropi: `{fitur_dict['glcm_entropy']:.2f}`")
                st.write(f"- Energi: `{fitur_dict['glcm_energy']:.4f}`")
                st.write(f"- Homogenitas: `{fitur_dict['glcm_homogeneity']:.4f}`")
            with c2:
                st.markdown("**Edge & Geometri:**")
                st.write(f"- Kepadatan Tepi: `{fitur_dict['edge_density']:.4f}`")
                st.write(f"- Jumlah Kontur: `{fitur_dict['num_contours']}`")
                st.write(f"- Entropi HOG: `{fitur_dict['hog_entropy']:.2f}`")
                st.write(f"- Rasio Orthogonal: `{fitur_dict['hog_ortho_ratio']:.3f}`")
            with c3:
                st.markdown("**Warna & SIFT Keypoints:**")
                st.write(f"- Mean RGB: `({fitur_dict['mean_r']:.1f}, {fitur_dict['mean_g']:.1f}, {fitur_dict['mean_b']:.1f})`")
                st.write(f"- SIFT Keypoints: `{fitur_dict['sift_num_keypoints']:.0f}`")
                st.write(f"- SIFT Spread: `{fitur_dict['sift_spatial_spread']:.1f}`")

if __name__ == '__main__':
    main()
