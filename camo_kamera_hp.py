"""
================================================================================
KLASIFIKASI CITRA PENATAAN SEPATU MENGGUNAKAN CAMO STUDIO (KAMERA HP)
METODE: EXTRA TREES CLASSIFIER (48 FITUR MULTIMODAL)
FITUR: LIVE REAL-TIME PROBABILITY + TEMPORAL SMOOTHING + SNAPSHOT [SPACE]
================================================================================
Aplikasi ini membaca stream langsung dari kamera HP melalui Camo Studio (Virtual Camera)
dengan sistem multithreading berkecepatan tinggi:
1. Video Preview Full 60 FPS (tanpa lag / patah-patah).
2. Live Tracking Probabilitas: Terus menghitung persentase secara real-time.
3. Temporal Smoothing (EMA): Menstabilkan fluktuasi noise sensor kamera agar
   angka akurasi tidak loncat-loncat saat objek diam.
4. Snapshot Kunci [SPACE]: Menjepret, membekukan hasil, dan menyimpan foto.

Kontrol Keyboard:
- [SPACE] : Kunci Hasil Jepretan (Freeze) / Buka Kembali Live Feed (Resume)
- [TAB]   : Beralih ke nomor indeks kamera lain (Camo Studio / Webcam)
- [S]     : Simpan foto jepretan saat ini ke folder 'tangkapan_kamera/'
- [Q/ESC] : Keluar dari aplikasi
================================================================================
"""

import os
import sys
import time
import threading
import cv2
import joblib
import numpy as np
from datetime import datetime
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# 1. MEMUAT BUNDLE MODEL EXTRA TREES CLASSIFIER
# ==============================================================================
def load_extratrees_bundle(bundle_path='model_extratrees_penataan_sepatu.pkl'):
    if not os.path.exists(bundle_path):
        print(f"[!] Error: File '{bundle_path}' tidak ditemukan.")
        sys.exit(1)
        
    print(f"[+] Memuat model Extra Trees dari '{bundle_path}'...")
    bundle = joblib.load(bundle_path)
    print(f"[+] Metode Aktif : {bundle.get('method_name', 'Extra Trees Classifier')}")
    print(f"[+] Akurasi CV   : {bundle.get('accuracy_cv', 0):.2%} | ROC AUC: {bundle.get('roc_auc', 0):.4f}")
    return bundle

# ==============================================================================
# 2. EKSTRAKSI 48 FITUR MULTIMODAL LENGKAP
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
# 3. DETEKSI KAMERA SISTEM
# ==============================================================================
def find_available_cameras():
    cams = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cams.append((idx, w, h))
            cap.release()
    return cams

# ==============================================================================
# 4. APLIKASI UTAMA LIVE TRACKING & SNAPSHOT
# ==============================================================================
def run_camo_app(default_cam_idx=None):
    os.makedirs('tangkapan_kamera', exist_ok=True)
    bundle = load_extratrees_bundle()
    
    model = bundle['model']
    scaler = bundle['scaler']
    kmeans_bovw = bundle['kmeans_bovw']
    fitur_cols = bundle['fitur_cols']
    optimal_th = bundle.get('optimal_threshold', 0.48)
    
    print("\n[+] Mendeteksi kamera pada sistem...")
    available_cams = find_available_cameras()
    if not available_cams:
        print("[!] Tidak ada kamera yang terdeteksi! Pastikan Camo Studio sudah terhubung.")
        return
        
    print("[+] Daftar Kamera Tersedia:")
    camo_candidate = 0
    for idx, w, h in available_cams:
        label = "Camo Studio (HD)" if (w >= 1280 or h >= 720) else "Kamera Standar"
        print(f"    - Index [{idx}] : {w}x{h} ({label})")
        if w >= 1280 and default_cam_idx is None:
            camo_candidate = idx
            
    if default_cam_idx is not None:
        cam_idx = default_cam_idx
    else:
        cam_idx = camo_candidate
        
    print(f"\n[+] Membuka kamera Index [{cam_idx}] (Camo Studio)...")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"[!] GAGAL membuka kamera Index {cam_idx}. Beralih ke index 0...")
        cam_idx = 0
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[!] Gagal membuka kamera. Periksa koneksi Camo Studio.")
            return

    win_name = "Klasifikasi Penataan Sepatu - Camo Studio (Live + Snapshot)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1080, 720)
    
    # --------------------------------------------------------------------------
    # THREADING BACKGROUND WORKER UNTUK PREDIKSI REAL-TIME TANPA LAG
    # --------------------------------------------------------------------------
    lock = threading.Lock()
    latest_frame = None
    is_running = True
    
    # Status Live Sharing Variables
    live_raw_prob_rapi = 0.5
    live_smooth_prob_rapi = 0.5
    live_latency = 0.0
    alpha_smoothing = 0.35 # Bobot Exponential Moving Average (Peredam Noise Sensor)
    
    def live_predictor_worker():
        nonlocal live_raw_prob_rapi, live_smooth_prob_rapi, live_latency
        while is_running:
            frame_to_process = None
            with lock:
                if latest_frame is not None:
                    frame_to_process = latest_frame.copy()
                    
            if frame_to_process is not None:
                t0 = time.time()
                try:
                    x_vec, _ = extract_48_features(frame_to_process, kmeans_bovw, fitur_cols)
                    x_scaled = scaler.transform(x_vec)
                    raw_prob = model.predict_proba(x_scaled)[0, 1]
                    dur = time.time() - t0
                    
                    with lock:
                        live_raw_prob_rapi = raw_prob
                        # Exponential Moving Average: menstabilkan fluktuasi pencahayaan kamera
                        live_smooth_prob_rapi = (alpha_smoothing * raw_prob) + ((1.0 - alpha_smoothing) * live_smooth_prob_rapi)
                        live_latency = dur
                except Exception as e:
                    pass
                    
            time.sleep(0.08) # Interval komputasi kontinu (~10 prediksi per detik)

    pred_thread = threading.Thread(target=live_predictor_worker, daemon=True)
    pred_thread.start()
    
    # Variabel Snapshot [SPACE]
    is_frozen = False
    frozen_frame = None
    frozen_header = ""
    frozen_detail = ""
    frozen_color = (0, 200, 0)
    
    print("\n" + "=" * 75)
    print(" SISTEM KLASIFIKASI LIVE AKTIF:")
    print("   - Persentase ditampilkan secara LIVE REAL-TIME di layar kamera.")
    print("   - Dilengkapi Temporal Stabilizer agar angka stabil saat objek diam.")
    print("   - Tekan [SPACE] : KUNCI JEPPRETAN (Freeze) & SIMPAN FOTO")
    print("   - Tekan [SPACE] lagi : Buka kembali Live Streaming")
    print("   - Tekan [TAB]   : Beralih ke kamera lain")
    print("   - Tekan [Q/ESC] : Keluar")
    print("=" * 75 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
            
        with lock:
            latest_frame = frame
            curr_smooth_prob = live_smooth_prob_rapi
            curr_raw_prob = live_raw_prob_rapi
            curr_latency = live_latency

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]: # Q or ESC
            break
        elif key == 9: # TAB -> Ganti kamera
            cap.release()
            cam_idx = (cam_idx + 1) % len(available_cams)
            print(f"[+] Beralih ke Kamera Index [{cam_idx}]...")
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            continue
        elif key == 32: # SPACE -> Toggle Freeze / Snapshot
            is_frozen = not is_frozen
            if is_frozen:
                # KUNCI JEPPRETAN
                frozen_frame = frame.copy()
                snap_is_rapi = curr_smooth_prob >= optimal_th
                snap_conf = (curr_smooth_prob if snap_is_rapi else (1.0 - curr_smooth_prob)) * 100
                snap_label = "PENATAAN RAPI" if snap_is_rapi else "PENATAAN BERANTAKAN"
                frozen_color = (0, 220, 0) if snap_is_rapi else (0, 0, 240)
                
                frozen_header = f"📸 [HASIL DIKUNCI] {snap_label} ({snap_conf:.1f}%)"
                frozen_detail = f"Prob Rapi: {curr_smooth_prob:.1%} | Prob Berantakan: {(1-curr_smooth_prob):.1%} | Th: {optimal_th:.2f}"
                
                # Simpan otomatis ke tangkapan_kamera/
                fname = f"tangkapan_kamera/jepret_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frozen_frame)
                print(f"[+] {frozen_header}")
                print(f"    Tersimpan ke: {fname}")
                print("    (Tekan [SPACE] lagi untuk melanjutkan live scanning)")
            else:
                print("[+] Mode Live Streaming dilanjutkan...")

        # ----------------------------------------------------------------------
        # RENDERING HUD PADA DISPLAY
        # ----------------------------------------------------------------------
        display_frame = frozen_frame.copy() if is_frozen else frame.copy()
        h, w, _ = display_frame.shape
        
        # Banner Transparan Atas (HUD)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, display_frame, 0.20, 0, display_frame)
        
        if is_frozen:
            # Tampilan Saat Terkunci (Snapshot Mode)
            cv2.putText(display_frame, frozen_header, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.05, frozen_color, 2)
            cv2.putText(display_frame, frozen_detail, (22, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1)
            cv2.putText(display_frame, "[FOTO DIKUNCI & TERSIMPAN] - TEKAN [SPACE] UNTUK KEMBALI LIVE", (w - 630, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 255), 1)
        else:
            # Tampilan Saat Live Scanning
            live_is_rapi = curr_smooth_prob >= optimal_th
            live_conf = (curr_smooth_prob if live_is_rapi else (1.0 - curr_smooth_prob)) * 100
            live_color = (0, 220, 0) if live_is_rapi else (0, 0, 240)
            live_label = "PENATAAN RAPI" if live_is_rapi else "PENATAAN BERANTAKAN"
            
            header_text = f"LIVE: {live_label} ({live_conf:.1f}%)"
            sub_text = f"Peluang Rapi: {curr_smooth_prob:.1%} | Peluang Berantakan: {(1-curr_smooth_prob):.1%} | Latensi: {curr_latency:.2f}s"
            
            cv2.circle(display_frame, (35, 40), 10, live_color, -1) # Titik deteksi
            cv2.putText(display_frame, header_text, (55, 48), cv2.FONT_HERSHEY_DUPLEX, 1.0, live_color, 2)
            cv2.putText(display_frame, sub_text, (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            
            # Bar Visual Live Probabilitas di Kanan Atas
            bar_w, bar_h = 240, 18
            bar_x, bar_y = w - bar_w - 25, 30
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
            fill_w = int(bar_w * curr_smooth_prob)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), live_color, -1)
            # Garis Ambang Batas (Threshold 0.48)
            th_x = int(bar_x + bar_w * optimal_th)
            cv2.line(display_frame, (th_x, bar_y - 4), (th_x, bar_y + bar_h + 4), (255, 255, 255), 2)
            cv2.putText(display_frame, f"Rapi: {curr_smooth_prob*100:.0f}%", (bar_x, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(display_frame, f"Th: {optimal_th:.2f}", (th_x - 18, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Banner Bawah (Footer Kontrol)
        cv2.rectangle(display_frame, (0, h - 35), (w, h), (15, 15, 15), -1)
        mode_str = "STATUS: TERKUNCI (FROZEN)" if is_frozen else "STATUS: LIVE SCANNING"
        footer_txt = f"{mode_str} | [SPACE] Jepret/Kunci | [TAB] Ganti Cam | [Q] Keluar"
        cv2.putText(display_frame, footer_txt, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)
        
        cv2.imshow(win_name, display_frame)
        
    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Aplikasi Camo Studio ditutup.")

if __name__ == '__main__':
    arg_idx = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        arg_idx = int(sys.argv[1])
    run_camo_app(arg_idx)
