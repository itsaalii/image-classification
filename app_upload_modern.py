"""
================================================================================
SHOESENSE AI - REACT 19 + TYPESCRIPT + VITE + TAILWIND CSS DASHBOARD SERVER
METODE: WEIGHTED SOFT VOTING ENSEMBLE (24 FITUR HANDCRAFTED: WARNA, TEKSTUR, TEPI, BENTUK)
================================================================================
Backend server modern berbasis Flask:
- Melayani REST API (/predict, /api/info, /api/health) dengan dukungan CORS penuh.
- Melayani SPA (Single Page Application) hasil kompilasi React 19 dari folder frontend/dist.
- Tanpa Streamlit (100% Native React 19 + TypeScript + Vite + Tailwind CSS).
================================================================================
"""

import os
import sys
import io
import time
import base64
import socket
import webbrowser
import threading
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# 1. MEMUAT BUNDLE MODEL WEIGHTED SOFT VOTING
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, 'model_terbaik_penataan_sepatu.pkl')

if not os.path.exists(BUNDLE_PATH):
    print(f"[!] File model '{BUNDLE_PATH}' tidak ditemukan!")
    sys.exit(1)

print(f"[+] Memuat model klasifikasi dari '{os.path.basename(BUNDLE_PATH)}'...")
bundle = joblib.load(BUNDLE_PATH)

model = bundle['model']
scaler = bundle['scaler']
fitur_cols = bundle['fitur_cols']
optimal_th = float(bundle.get('optimal_threshold', 0.44))
acc_cv = float(bundle.get('accuracy', bundle.get('accuracy_cv', 0.9783)))
roc_auc = float(bundle.get('roc_auc', 0.9976))

print(f"[+] Model Aktif   : Weighted Soft Voting Ensemble ({len(fitur_cols)} Fitur Handcrafted)")
print(f"[+] Komposisi     : 9 Warna RGB + 5 GLCM CLAHE + 4 Tepi Struktural + 6 Bentuk Kontur")
print(f"[+] Akurasi CV    : {acc_cv:.2%} | ROC AUC: {roc_auc:.4f} | Threshold: {optimal_th:.2f}")

# ==============================================================================
# 2. EKSTRAKSI 24 FITUR HANDCRAFTED (WARNA, TEKSTUR GLCM, TEPI, BENTUK)
#    (100% SAMA PERSIS DENGAN PIPELINE TRAINING DI klasifikasi.ipynb)
# ==============================================================================
def extract_24_features(img_bgr):
    # Standardisasi resolusi input menjadi 512x512 piksel (konsisten dengan dataset training)
    img_512 = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_512, cv2.COLOR_BGR2GRAY)
    img_res = cv2.resize(img_gray, (256, 256))
    
    # 1. 9 Fitur Warna RGB (Mean, Std, Skewness)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    fitur_dict = {
        'mean_r': float(np.mean(r)), 'mean_g': float(np.mean(g)), 'mean_b': float(np.mean(b)),
        'std_r': float(np.std(r)), 'std_g': float(np.std(g)), 'std_b': float(np.std(b)),
        'skew_r': float(skew(r.ravel())), 'skew_g': float(skew(g.ravel())), 'skew_b': float(skew(b.ravel()))
    }
    
    # 2. 5 Fitur Tekstur GLCM dengan Prapemrosesan CLAHE (256x256)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_res)
    glcm = graycomatrix(img_clahe, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    fitur_dict['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
    p_norm = glcm / (np.sum(glcm) + 1e-15)
    fitur_dict['glcm_entropy'] = float(-np.sum(p_norm * np.log2(p_norm + 1e-15)))
    fitur_dict['glcm_correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
    fitur_dict['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
    fitur_dict['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
    
    # 3. 4 Fitur Tepi Struktural (Canny, Sobel, Laplacian)
    canny = cv2.Canny(img_clahe, 50, 150)
    fitur_dict['edge_density'] = float(np.mean(canny > 0))
    sobelx = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_clahe, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    fitur_dict['sobel_mean'] = float(np.mean(sobel_mag))
    fitur_dict['sobel_std'] = float(np.std(sobel_mag))
    laplacian = cv2.Laplacian(img_clahe, cv2.CV_64F)
    fitur_dict['laplacian_var'] = float(np.var(laplacian))
    
    # 4. 6 Fitur Geometri Bentuk (Kontur & Invarian Momen Hu)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fitur_dict['num_contours'] = float(len(contours))
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
    fitur_dict['contour_area_mean'] = float(np.mean(areas)) if len(areas) > 0 else 0.0
    fitur_dict['contour_area_std'] = float(np.std(areas)) if len(areas) > 0 else 0.0
    solidities = [
        cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c))
        for c in contours
        if cv2.contourArea(c) > 20 and cv2.contourArea(cv2.convexHull(c)) > 0
    ]
    fitur_dict['contour_solidity'] = float(np.mean(solidities)) if len(solidities) > 0 else 0.0
    hu = cv2.HuMoments(cv2.moments(canny)).flatten()
    fitur_dict['hu_moment_1'] = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-15))
    fitur_dict['hu_moment_2'] = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-15))
    
    # Menyusun vektor fitur persis sesuai urutan kolom pada model
    x_vec = np.array([[fitur_dict[c] for c in fitur_cols]])
    return x_vec, fitur_dict

# Alias kompatibilitas
extract_38_features = extract_24_features
extract_features = extract_24_features

# ==============================================================================
# 3. KONFIGURASI FLASK SERVER & STATIC ROUTING (REACT 19 SPA)
# ==============================================================================
FRONTEND_DIST = os.path.join(BASE_DIR, 'frontend', 'dist')
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path='')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'server_time': time.time(),
        'model_loaded': True
    })

@app.route('/api/info', methods=['GET'])
def model_info():
    return jsonify({
        'nama_model': 'Weighted Soft Voting Ensemble (SVM:2, ET:2, GB:1)',
        'akurasi': acc_cv,
        'roc_auc': roc_auc,
        'threshold': optimal_th,
        'total_fitur': len(fitur_cols),
        'fitur_cols': fitur_cols,
        'kategori_fitur': {
            'rgb': 9,
            'glcm': 5,
            'edge': 4,
            'shape': 6
        }
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    start_time = time.time()
    img_bgr = None

    # Skenario 1: File multipart
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Skenario 2: Base64 JSON
    elif request.is_json and 'image' in request.json:
        b64_str = request.json['image']
        if ',' in b64_str:
            b64_str = b64_str.split(',', 1)[1]
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({'error': 'Citra tidak ditemukan atau format tidak didukung (harus JPG/PNG/WEBP)'}), 400

    try:
        # Ekstraksi 24 Fitur Handcrafted (Warna RGB, GLCM CLAHE, Tepi Struktural, Bentuk Kontur)
        x_vec, fitur_dict = extract_24_features(img_bgr)
        x_scaled = scaler.transform(x_vec)

        # Prediksi Probabilitas Ansambel
        prob_rapi = float(model.predict_proba(x_scaled)[0, 1])
        prob_berantakan = 1.0 - prob_rapi
        is_rapi = bool(prob_rapi >= optimal_th)
        confidence = float((prob_rapi if is_rapi else prob_berantakan) * 100.0)

        # Buat Thumbnail Base64 untuk pratinjau cepat
        h, w = img_bgr.shape[:2]
        scale = min(240 / w, 240 / h)
        img_thumb = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', img_thumb)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        elapsed_ms = round((time.time() - start_time) * 1000)

        return jsonify({
            'is_rapi': is_rapi,
            'prediksi': 'rapi' if is_rapi else 'berantakan',
            'confidence': confidence,
            'prob_rapi': prob_rapi,
            'prob_berantakan': prob_berantakan,
            'threshold': optimal_th,
            'features': fitur_dict,
            'image_b64': img_b64,
            'inference_time_ms': elapsed_ms
        })
    except Exception as e:
        return jsonify({'error': f'Gagal mengekstrak fitur atau inferensi: {str(e)}'}), 500

# Routing Single Page Application (React 19)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    if path != "" and os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    elif os.path.exists(os.path.join(FRONTEND_DIST, 'index.html')):
        return send_from_directory(FRONTEND_DIST, 'index.html')
    else:
        return (
            """
            <html>
                <head><title>ShoeSense AI - Build Pending</title></head>
                <body style="font-family:sans-serif; text-align:center; padding:50px; background:#0f172a; color:#f8fafc;">
                    <h2>Folder kompilasi React 'frontend/dist' belum ditemukan</h2>
                    <p>Silakan jalankan perintah berikut di folder proyek untuk mengompilasi frontend:</p>
                    <pre style="background:#1e293b; padding:15px; border-radius:8px; display:inline-block;">cd frontend && npm install && npm run build</pre>
                </body>
            </html>
            """,
            404
        )

# ==============================================================================
# 4. PEMILIHAN PORT & LAUNCHER
# ==============================================================================
def find_available_port(start_port=5000, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

def run_app():
    port = find_available_port(5000)
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 75)
    print(" 👟 SHOESENSE AI - REACT 19 + TYPESCRIPT + VITE + TAILWIND CSS")
    print(f" [+] Server berjalan di : {url}")
    print(f" [+] Frontend Build     : {FRONTEND_DIST}")
    print(" [+] Membuka antarmuka otomatis di browser default Anda...")
    print("=" * 75 + "\n")

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False)

if __name__ == '__main__':
    run_app()
