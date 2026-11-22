"""
================================================================================
REAL-TIME DESKTOP CLASSIFIER DENGAN STREAM KAMERA HP (OPENCV)
================================================================================
Menggunakan stream video real-time dari aplikasi kamera HP seperti:
1. IP Webcam (Android) -> URL bawaan: http://<IP_HP>:8080/video
2. DroidCam (Android/iOS) -> URL bawaan: http://<IP_HP>:4747/video
3. Webcam bawaan laptop (Device Index: 0)

Kontrol Keyboard:
- [SPACE] : Lakukan klasifikasi penataan sepatu pada frame saat ini
- [A]     : Nyalakan/Matikan mode Klasifikasi Otomatis Tiap Detik
- [Q]     : Keluar dari aplikasi
"""

import sys
import time
import cv2
import joblib
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

def load_bundle(bundle_path='model_terbaik_penataan_sepatu.pkl'):
    print(f"[+] Memuat model bundle dari '{bundle_path}'...")
    bundle = joblib.load(bundle_path)
    print(f"[+] Model berhasil dimuat! Akurasi evaluasi: {bundle.get('accuracy', 0):.2%}")
    return bundle

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
    return x_vec

def run_camera_stream(source=0):
    bundle = load_bundle()
    model = bundle['model']
    scaler = bundle['scaler']
    fitur_cols = bundle['fitur_cols']
    optimal_th = bundle.get('optimal_threshold', 0.57)
    kmeans_bovw = bundle['kmeans_bovw']
    
    print(f"\n[+] Membuka stream kamera: {source}...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[!] GAGAL membuka stream kamera dari: {source}")
        print("[!] Pastikan kamera aktif dan URL IP Webcam benar.")
        return
        
    last_status = "MENUNGGU JEPPRETAN [SPACE]"
    last_color = (200, 200, 200)
    auto_classify = False
    last_auto_time = 0
    
    cv2.namedWindow("Klasifikasi Penataan Sepatu - Kamera HP", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Klasifikasi Penataan Sepatu - Kamera HP", 800, 600)
    
    print("\n[+] KAMERA AKTIF!")
    print("    - Tekan [SPACE] : Klasifikasi frame ini")
    print("    - Tekan [A]     : Toggle klasifikasi otomatis")
    print("    - Tekan [Q]     : Keluar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Kehilangan sinyal video...")
            break
            
        now = time.time()
        do_classify = False
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == 32: # SPACE
            do_classify = True
        elif key == ord('a') or key == ord('A'):
            auto_classify = not auto_classify
            print(f"[+] Mode Auto-Klasifikasi: {'AKTIF' if auto_classify else 'MATI'}")
            
        if auto_classify and (now - last_auto_time > 1.2):
            do_classify = True
            last_auto_time = now
            
        if do_classify:
            t0 = time.time()
            x_vec = extract_48_features(frame, kmeans_bovw, fitur_cols)
            x_scaled = scaler.transform(x_vec)
            prob_rapi = model.predict_proba(x_scaled)[0, 1]
            dur = time.time() - t0
            
            if prob_rapi >= optimal_th:
                last_status = f"STATUS: RAPI ({prob_rapi:.1%}) [{dur:.2f}s]"
                last_color = (0, 200, 0) # Hijau
            else:
                last_status = f"STATUS: BERANTAKAN ({(1-prob_rapi):.1%}) [{dur:.2f}s]"
                last_color = (0, 0, 220) # Merah
                
            print(f"[{time.strftime('%H:%M:%S')}] {last_status}")
            
        # Gambar Overlay HUD
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.putText(frame, last_status, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, last_color, 2)
        
        info_txt = f"Auto: {'ON' if auto_classify else 'OFF'} | [SPACE] Jepret | [A] Auto | [Q] Keluar"
        cv2.putText(frame, info_txt, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        
        cv2.imshow("Klasifikasi Penataan Sepatu - Kamera HP", frame)
        
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Stream kamera ditutup.")

if __name__ == '__main__':
    source_arg = 0
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        if source_arg.isdigit():
            source_arg = int(source_arg)
    run_camera_stream(source_arg)
