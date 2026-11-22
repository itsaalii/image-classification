"""
================================================================================
SHOESENSE AI - MODERN DRIBBBLE-STYLE WEB DASHBOARD (UPLOAD MODE)
METODE: EXTRA TREES CLASSIFIER (48 FITUR MULTIMODAL HANDCRAFTED)
================================================================================
Desain terinspirasi langsung dari UI Mockup Dribbble:
- Floating Tablet Frame dengan Dark Left Dock Sidebar (#16161A).
- Top Bento Row: Portfolio Card (Hasil Klasifikasi & Preview) + 3 Pastel Cards (GLCM, Edge, SIFT).
- Bottom Bento Row: Feature Table (48 Fitur) + Dark Promo Hero Card ("Extra Trees Engine").
- 100% Interaktif: Drag & Drop upload, animasi halus, pembaruan instan via AJAX.
================================================================================
"""

import os
import sys
import io
import base64
import socket
import webbrowser
import threading
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# 1. MEMUAT BUNDLE MODEL WEIGHTED SOFT VOTING
# ==============================================================================
BUNDLE_PATH = 'model_terbaik_penataan_sepatu.pkl' if os.path.exists('model_terbaik_penataan_sepatu.pkl') else 'model_extratrees_penataan_sepatu.pkl'

def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    try:
        return joblib.load(BUNDLE_PATH)
    except Exception as e:
        print(f"Error memuat bundle: {e}")
        return None

bundle = load_bundle()
if bundle is None:
    print(f"[!] File model '{BUNDLE_PATH}' tidak ditemukan!")
    sys.exit(1)

model = bundle['model']
scaler = bundle['scaler']
kmeans_bovw = bundle['kmeans_bovw']
fitur_cols = bundle['fitur_cols']
optimal_th = bundle.get('optimal_threshold', 0.57)
acc_cv = bundle.get('accuracy', bundle.get('accuracy_cv', 0.9267))
roc_auc = bundle.get('roc_auc', 0.9663)

# ==============================================================================
# 2. EKSTRAKSI 48 FITUR MULTIMODAL
# ==============================================================================
def extract_48_features(img_bgr):
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
    
    # 5. 14 Fitur SIFT & BoVW (Fitur HOG telah ditiadakan)
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
# 3. HTML + CSS TEMPLATE (EXACT DRIBBBLE MOCKUP REPLICA)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShoeSense AI - Overview Dashboard</title>
    <!-- Google Fonts Plus Jakarta Sans -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: #E8F1FC;
            background-image: 
                radial-gradient(at 10% 20%, rgba(219, 234, 254, 0.8) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(224, 231, 255, 0.8) 0px, transparent 50%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 30px 20px;
            color: #1A1D26;
        }

        /* Floating Tablet Frame */
        .tablet-device {
            width: 100%;
            max-width: 1220px;
            background: #141416;
            border-radius: 44px;
            padding: 10px;
            box-shadow: 0 35px 80px -15px rgba(15, 23, 42, 0.25);
            display: flex;
            position: relative;
        }

        /* Left Dock Sidebar */
        .sidebar-dock {
            width: 82px;
            background: #141416;
            border-radius: 34px 0 0 34px;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 0;
            flex-shrink: 0;
        }

        .logo-box {
            width: 48px;
            height: 48px;
            background: #FDF4E3;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 40px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
        }

        .logo-box:hover {
            transform: scale(1.05);
        }

        .nav-icons {
            display: flex;
            flex-direction: column;
            gap: 28px;
            align-items: center;
            width: 100%;
        }

        .nav-item {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #71717A;
            cursor: pointer;
            transition: all 0.2s;
        }

        .nav-item:hover, .nav-item.active {
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.1);
        }

        .nav-item svg {
            width: 22px;
            height: 22px;
            stroke-width: 2;
        }

        .dock-footer {
            margin-top: auto;
        }

        /* Main Canvas Container */
        .main-canvas {
            flex: 1;
            background: #FFFFFF;
            border-radius: 34px;
            padding: 32px 36px;
            display: flex;
            flex-direction: column;
            gap: 26px;
            overflow: hidden;
        }

        /* Header Bar */
        .header-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-left h1 {
            font-size: 28px;
            font-weight: 800;
            color: #11142D;
            letter-spacing: -0.5px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .icon-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            color: #11142D;
            position: relative;
            transition: background 0.2s;
        }

        .icon-circle:hover {
            background: #F4F5F7;
        }

        .icon-circle svg {
            width: 20px;
            height: 20px;
        }

        .badge-dot {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 7px;
            height: 7px;
            background: #FF5A5F;
            border-radius: 50%;
        }

        .user-pill {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 4px 12px 4px 4px;
            border-radius: 30px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .user-pill:hover {
            background: #F4F5F7;
        }

        .user-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #FFE5D9;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .user-name {
            font-size: 14px;
            font-weight: 700;
            color: #11142D;
        }

        /* Layout Grid */
        .bento-row-top {
            display: grid;
            grid-template-columns: 1.15fr 1.35fr;
            gap: 24px;
        }

        /* Section Headings */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 800;
            color: #11142D;
        }

        .section-action {
            cursor: pointer;
            color: #11142D;
        }

        /* Portfolio Big Card */
        .portfolio-card {
            background: #F0F6FE;
            border-radius: 26px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            min-height: 215px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.02);
            transition: all 0.3s ease;
        }

        .port-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .port-balance-val {
            font-size: 27px;
            font-weight: 800;
            color: #11142D;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }

        .port-balance-sub {
            font-size: 13px;
            font-weight: 500;
            color: #808191;
        }

        .menu-dots {
            color: #A0AEC0;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
        }

        /* Sparkline Chart & Preview Area */
        .chart-preview-container {
            display: flex;
            align-items: center;
            gap: 16px;
            margin: 10px 0;
            position: relative;
        }

        .preview-thumb {
            width: 74px;
            height: 74px;
            border-radius: 18px;
            object-fit: cover;
            border: 2px solid #FFFFFF;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            display: none;
        }

        .chart-wrapper {
            flex: 1;
            position: relative;
            height: 65px;
        }

        .floating-tag {
            position: absolute;
            top: 2px;
            left: 55%;
            background: #11142D;
            color: #FFFFFF;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
        }

        .time-pills {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .time-pill {
            font-size: 11px;
            font-weight: 700;
            color: #808191;
            padding: 4px 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .time-pill.active, .time-pill:hover {
            color: #11142D;
            background: rgba(255, 255, 255, 0.8);
        }

        /* 3 Pastel Cards Container */
        .pastel-cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }

        .pastel-card {
            border-radius: 24px;
            padding: 20px 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 215px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .pastel-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.04);
        }

        /* 3 Exact Colors from Mockup */
        .card-purple {
            background: #EFEAFE; /* Soft Lavender */
        }
        .card-green {
            background: #E6F7EC; /* Soft Sage / Mint */
        }
        .card-yellow {
            background: #FEF6E1; /* Soft Cream / Peach */
        }

        .card-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .asset-title {
            font-size: 16px;
            font-weight: 800;
            color: #11142D;
            margin-bottom: 4px;
        }

        .asset-sub {
            font-size: 12px;
            font-weight: 600;
            color: #6C727F;
        }

        .card-bottom {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .crypto-icon-box {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 800;
            color: #11142D;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        }

        .rate-pill {
            font-size: 11px;
            font-weight: 700;
            color: #059669;
            background: rgba(255, 255, 255, 0.75);
            padding: 4px 8px;
            border-radius: 12px;
        }

        /* Bottom Row */
        .bento-row-bottom {
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 24px;
        }

        /* Table Section */
        .table-card {
            background: #FFFFFF;
            border-radius: 26px;
            display: flex;
            flex-direction: column;
        }

        .table-filter-bar {
            display: flex;
            gap: 10px;
        }

        .filter-pill {
            font-size: 12px;
            font-weight: 700;
            color: #11142D;
            background: #F4F5F7;
            padding: 6px 14px;
            border-radius: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .features-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
        }

        .features-table th {
            text-align: left;
            font-size: 11px;
            font-weight: 600;
            color: #9A9FA5;
            padding-bottom: 12px;
            border-bottom: 1px solid #F4F5F7;
        }

        .features-table td {
            padding: 12px 0;
            font-size: 13px;
            color: #11142D;
            border-bottom: 1px solid #FAFAFB;
        }

        .feature-item-col {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .feature-icon-badge {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: #11142D;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 800;
            flex-shrink: 0;
        }

        .feature-name-box h4 {
            font-size: 13px;
            font-weight: 700;
            color: #11142D;
        }

        .feature-name-box span {
            font-size: 11px;
            color: #808191;
        }

        .val-pill {
            font-weight: 700;
            color: #10B981;
            font-size: 12px;
        }

        .star-icon {
            color: #D1D5DB;
            cursor: pointer;
            font-size: 14px;
            text-align: right;
        }

        /* Black Promo Hero Card */
        .promo-card {
            background: #11142D;
            border-radius: 28px;
            padding: 30px;
            color: #FFFFFF;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 250px;
            box-shadow: 0 15px 30px rgba(17, 20, 45, 0.15);
        }

        .promo-title {
            font-size: 26px;
            font-weight: 800;
            line-height: 1.25;
            max-width: 250px;
        }

        .pill-highlight {
            display: inline-block;
            border: 1.5px solid #FFFFFF;
            border-radius: 20px;
            padding: 0 10px;
            font-weight: 700;
        }

        .promo-desc {
            font-size: 12px;
            color: #A0AEC0;
            margin-top: 10px;
            max-width: 260px;
            line-height: 1.5;
        }

        .btn-upload-pill {
            background: #FFFFFF;
            color: #11142D;
            border: none;
            padding: 12px 26px;
            border-radius: 24px;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
            width: fit-content;
            margin-top: 20px;
            z-index: 2;
        }

        .btn-upload-pill:hover {
            background: #F0F6FE;
            transform: scale(1.03);
        }

        /* Abstract Line Graphics */
        .promo-art {
            position: absolute;
            bottom: -20px;
            right: -20px;
            width: 220px;
            height: 220px;
            pointer-events: none;
            opacity: 0.35;
        }

        /* Hidden File Input */
        #fileInput {
            display: none;
        }

        /* Status Colors */
        .status-rapi {
            color: #059669 !important;
        }
        .status-berantakan {
            color: #DC2626 !important;
        }

        /* Responsive Media Queries */
        @media (max-width: 1024px) {
            .bento-row-top, .bento-row-bottom {
                grid-template-columns: 1fr;
            }
            .tablet-device {
                flex-direction: column;
            }
            .sidebar-dock {
                width: 100%;
                height: 60px;
                flex-direction: row;
                border-radius: 34px 34px 0 0;
                padding: 0 20px;
            }
            .nav-icons {
                flex-direction: row;
                margin-left: 20px;
            }
            .logo-box {
                margin-bottom: 0;
            }
        }
    </style>
</head>
<body>

    <!-- Floating Tablet Frame -->
    <div class="tablet-device">

        <!-- Left Dock Sidebar -->
        <aside class="sidebar-dock">
            <div class="logo-box" title="ShoeSense AI" onclick="triggerUpload()">
                👟
            </div>

            <div class="nav-icons">
                <div class="nav-item active" title="Overview">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <rect x="3" y="3" width="7" height="7" rx="2"></rect>
                        <rect x="14" y="3" width="7" height="7" rx="2"></rect>
                        <rect x="14" y="14" width="7" height="7" rx="2"></rect>
                        <rect x="3" y="14" width="7" height="7" rx="2"></rect>
                    </svg>
                </div>
                <div class="nav-item" title="Unggah Foto" onclick="triggerUpload()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                </div>
                <div class="nav-item" title="Statistik & Analisis">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                </div>
                <div class="nav-item" title="Model Extra Trees">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                        <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                        <line x1="12" y1="22.08" x2="12" y2="12"></line>
                    </svg>
                </div>
                <div class="nav-item" title="Profil Pengguna">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                        <circle cx="12" cy="7" r="4"></circle>
                    </svg>
                </div>
            </div>

            <div class="dock-footer">
                <div class="nav-item" title="Keluar">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                        <polyline points="16 17 21 12 16 7"></polyline>
                        <line x1="21" y1="12" x2="9" y2="12"></line>
                    </svg>
                </div>
            </div>
        </aside>

        <!-- Main Canvas Content -->
        <main class="main-canvas">

            <!-- Top Header -->
            <header class="header-bar">
                <div class="header-left">
                    <h1>Overview</h1>
                </div>
                <div class="header-right">
                    <div class="icon-circle" title="Pencarian">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                    </div>
                    <div class="icon-circle" title="Notifikasi">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                            <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                        </svg>
                        <div class="badge-dot"></div>
                    </div>
                    <div class="user-pill">
                        <div class="user-avatar">🧑‍💻</div>
                        <span class="user-name">Data Scientist</span>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </div>
                </div>
            </header>

            <!-- Bento Top Row -->
            <section class="bento-row-top">

                <!-- Left: Portfolio Card (Classification Result) -->
                <div>
                    <div class="section-header">
                        <span class="section-title">Portfolio</span>
                    </div>

                    <div class="portfolio-card" id="portfolioCard">
                        <div class="port-header">
                            <div>
                                <div class="port-balance-val" id="resClassTitle">Siap Unggah Foto</div>
                                <div class="port-balance-sub" id="resSubtitle">Klik tombol 'Unggah Foto' untuk memulai klasifikasi</div>
                            </div>
                            <div class="menu-dots" onclick="triggerUpload()">⋮</div>
                        </div>

                        <!-- Sparkline & Image Preview Area -->
                        <div class="chart-preview-container">
                            <img id="imgPreview" class="preview-thumb" alt="Preview Sepatu">
                            
                            <div class="chart-wrapper">
                                <!-- Modern Smooth SVG Wave -->
                                <svg width="100%" height="65" viewBox="0 0 320 65" fill="none" preserveAspectRatio="none">
                                    <path d="M0 45 C 40 42, 70 48, 110 44 C 150 40, 180 43, 210 38 C 230 35, 240 22, 250 25 L 250 15 L 255 15 L 255 45 C 270 40, 290 44, 320 30" 
                                          stroke="#38BDF8" stroke-width="2.5" fill="none"/>
                                    <circle id="chartPinDot" cx="252" cy="15" r="4.5" fill="#38BDF8" stroke="#FFFFFF" stroke-width="2"/>
                                    <line x1="252" y1="15" x2="252" y2="65" stroke="#38BDF8" stroke-dasharray="2 2" stroke-width="1.5" opacity="0.6"/>
                                </svg>
                                <div class="floating-tag" id="floatingTag">Akurasi 92.3%</div>
                            </div>
                        </div>

                        <!-- Time / Feature Pills -->
                        <div class="time-pills">
                            <span class="time-pill">1H</span>
                            <span class="time-pill">24H</span>
                            <span class="time-pill active">1W</span>
                            <span class="time-pill">1M</span>
                            <span class="time-pill">1Y</span>
                            <span class="time-pill">ALL</span>
                        </div>
                    </div>
                </div>

                <!-- Right: 3 Pastel Cards (Your Assets) -->
                <div>
                    <div class="section-header">
                        <span class="section-title">Your Assets</span>
                        <span class="section-action">⇄</span>
                    </div>

                    <div class="pastel-cards-grid">
                        <!-- Card 1: Lavender (GLCM) -->
                        <div class="pastel-card card-purple">
                            <div class="card-top">
                                <div>
                                    <div class="asset-title" id="valGlcmMain">0.842 GLCM</div>
                                    <div class="asset-sub" id="valGlcmSub">Homogenitas</div>
                                </div>
                                <div class="menu-dots">⋮</div>
                            </div>
                            <div class="card-bottom">
                                <div class="crypto-icon-box">🔲</div>
                                <div class="rate-pill" id="pillGlcm">+ 0.14%</div>
                            </div>
                        </div>

                        <!-- Card 2: Green Sage (Edge & Laplacian) -->
                        <div class="pastel-card card-green">
                            <div class="card-top">
                                <div>
                                    <div class="asset-title" id="valEdgeMain">428.0 Var</div>
                                    <div class="asset-sub" id="valEdgeSub">Laplacian Ketajaman</div>
                                </div>
                                <div class="menu-dots">⋮</div>
                            </div>
                            <div class="card-bottom">
                                <div class="crypto-icon-box">📐</div>
                                <div class="rate-pill" id="pillEdge">+ 0.31%</div>
                            </div>
                        </div>

                        <!-- Card 3: Yellow Cream (SIFT & HOG) -->
                        <div class="pastel-card card-yellow">
                            <div class="card-top">
                                <div>
                                    <div class="asset-title" id="valSiftMain">164 Pts</div>
                                    <div class="asset-sub" id="valSiftSub">SIFT Keypoints</div>
                                </div>
                                <div class="menu-dots">⋮</div>
                            </div>
                            <div class="card-bottom">
                                <div class="crypto-icon-box">🧭</div>
                                <div class="rate-pill" id="pillSift">+ 0.27%</div>
                            </div>
                        </div>
                    </div>
                </div>

            </section>

            <!-- Bento Bottom Row -->
            <section class="bento-row-bottom">

                <!-- Left: Market / Feature Extraction Table -->
                <div class="table-card">
                    <div class="section-header">
                        <span class="section-title" id="tableHeaderTitle">Analisis 48 Fitur Multimodal</span>
                        <div class="table-filter-bar">
                            <span class="filter-pill">24h ▾</span>
                            <span class="filter-pill">Top gainers ▾</span>
                        </div>
                    </div>

                    <table class="features-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Value</th>
                                <th>Weight</th>
                                <th>Domain</th>
                                <th style="text-align: right;">Watch</th>
                            </tr>
                        </thead>
                        <tbody id="featuresTbody">
                            <tr>
                                <td>
                                    <div class="feature-item-col">
                                        <div class="feature-icon-badge">G</div>
                                        <div class="feature-name-box">
                                            <h4>GLCM Contrast</h4>
                                            <span>Slide 19-20 CLAHE</span>
                                        </div>
                                    </div>
                                </td>
                                <td id="rowGlcmVal">184.20</td>
                                <td><span class="val-pill" id="rowGlcmCont">+13.38%</span></td>
                                <td>Tekstur</td>
                                <td class="star-icon">☆</td>
                            </tr>
                            <tr>
                                <td>
                                    <div class="feature-item-col">
                                        <div class="feature-icon-badge">E</div>
                                        <div class="feature-name-box">
                                            <h4>Canny Edge Density</h4>
                                            <span>Threshold 50-150</span>
                                        </div>
                                    </div>
                                </td>
                                <td id="rowEdgeVal">0.0842</td>
                                <td><span class="val-pill" id="rowEdgeCont">+11.19%</span></td>
                                <td>Garis Tepi</td>
                                <td class="star-icon">☆</td>
                            </tr>
                            <tr>
                                <td>
                                    <div class="feature-item-col">
                                        <div class="feature-icon-badge">S</div>
                                        <div class="feature-name-box">
                                            <h4>SIFT Visual Words</h4>
                                            <span>Bag of Visual Words</span>
                                        </div>
                                    </div>
                                </td>
                                <td id="rowSiftVal">164 pts</td>
                                <td><span class="val-pill" id="rowSiftCont">+7.57%</span></td>
                                <td>Keypoints</td>
                                <td class="star-icon">☆</td>
                            </tr>
                            <tr>
                                <td>
                                    <div class="feature-item-col">
                                        <div class="feature-icon-badge">H</div>
                                        <div class="feature-name-box">
                                            <h4>HOG Ortho Ratio</h4>
                                            <span>Arah Gradien 90°</span>
                                        </div>
                                    </div>
                                </td>
                                <td id="rowHogVal">0.4120</td>
                                <td><span class="val-pill" id="rowHogCont">+6.80%</span></td>
                                <td>Orientasi</td>
                                <td class="star-icon">☆</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Right: Black Promo Card ("Earn free crypto" style) -->
                <div class="promo-card">
                    <div>
                        <div class="promo-title">
                            Uji <span class="pill-highlight">instan</span> model Extra Trees!
                        </div>
                        <div class="promo-desc">
                            Klasifikasi keteraturan penataan sepatu murni Machine Learning Klasikal (Non-NN) dengan 48 fitur multimodal.
                        </div>
                    </div>

                    <button class="btn-upload-pill" onclick="triggerUpload()">
                        Unggah Foto Sekarang
                    </button>

                    <!-- Abstract Wireframe Art -->
                    <svg class="promo-art" viewBox="0 0 200 200" fill="none" stroke="#FFFFFF" stroke-width="1.2">
                        <polygon points="30,120 150,40 180,140 60,180" opacity="0.6"/>
                        <polygon points="50,140 170,60 190,160 80,195" opacity="0.4"/>
                        <line x1="30" y1="120" x2="80" y2="195" opacity="0.5"/>
                        <line x1="150" y1="40" x2="190" y2="160" opacity="0.5"/>
                    </svg>
                </div>

            </section>

        </main>
    </div>

    <!-- Hidden Input File -->
    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp,image/bmp">

    <script>
        const fileInput = document.getElementById('fileInput');

        function triggerUpload() {
            fileInput.click();
        }

        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                uploadImageFile(this.files[0]);
            }
        });

        // Drag & Drop pada Tablet Device
        const dropZone = document.querySelector('.tablet-device');
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.boxShadow = "0 0 30px rgba(56, 189, 248, 0.4)";
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.boxShadow = "";
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.boxShadow = "";
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                uploadImageFile(e.dataTransfer.files[0]);
            }
        });

        function uploadImageFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            // Tampilkan status proses
            document.getElementById('resClassTitle').innerText = "Menganalisis...";
            document.getElementById('resSubtitle').innerText = "Sedang mengekstrak 48 fitur multimodal...";

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                updateDashboard(data);
            })
            .catch(err => {
                console.error(err);
                alert('Gagal mengirim gambar untuk analisis.');
            });
        }

        function updateDashboard(data) {
            const isRapi = data.is_rapi;
            const conf = data.confidence.toFixed(1);
            const probRapi = (data.prob_rapi * 100).toFixed(1);
            const probBerantakan = (data.prob_berantakan * 100).toFixed(1);

            // 1. Update Portfolio Card
            const titleEl = document.getElementById('resClassTitle');
            if (isRapi) {
                titleEl.innerHTML = `<span class="status-rapi">🟢 PENATAAN RAPI</span>`;
                document.getElementById('resSubtitle').innerText = `Keyakinan: ${conf}% | Prob: ${probRapi}% (Threshold: ${data.threshold})`;
                document.getElementById('floatingTag').innerText = `Rapi: ${probRapi}%`;
                document.getElementById('floatingTag').style.background = "#059669";
            } else {
                titleEl.innerHTML = `<span class="status-berantakan">🔴 PENATAAN BERANTAKAN</span>`;
                document.getElementById('resSubtitle').innerText = `Keyakinan: ${conf}% | Prob: ${probBerantakan}% (Threshold: ${data.threshold})`;
                document.getElementById('floatingTag').innerText = `Acak: ${probBerantakan}%`;
                document.getElementById('floatingTag').style.background = "#DC2626";
            }

            // Preview Thumbnail
            if (data.image_b64) {
                const imgEl = document.getElementById('imgPreview');
                imgEl.src = "data:image/jpeg;base64," + data.image_b64;
                imgEl.style.display = "block";
            }

            // 2. Update 3 Pastel Cards
            const f = data.features;
            
            // GLCM
            document.getElementById('valGlcmMain').innerText = `${f.glcm_homogeneity.toFixed(3)} GLCM`;
            document.getElementById('valGlcmSub').innerText = `Kontras: ${f.glcm_contrast.toFixed(1)}`;
            document.getElementById('pillGlcm').innerText = isRapi ? "+ 0.42%" : "- 0.65%";

            // Edge
            document.getElementById('valEdgeMain').innerText = `${f.laplacian_var.toFixed(1)} Var`;
            document.getElementById('valEdgeSub').innerText = `Canny Dens: ${f.edge_density.toFixed(3)}`;
            document.getElementById('pillEdge').innerText = `Th: ${data.threshold}`;

            // SIFT
            document.getElementById('valSiftMain').innerText = `${f.sift_num_keypoints.toFixed(0)} Pts`;
            document.getElementById('valSiftSub').innerText = `HOG Ent: ${f.hog_entropy.toFixed(2)}`;
            document.getElementById('pillSift').innerText = "150 Pohon";

            // 3. Update Table
            document.getElementById('rowGlcmVal').innerText = f.glcm_contrast.toFixed(2);
            document.getElementById('rowEdgeVal').innerText = f.edge_density.toFixed(4);
            document.getElementById('rowSiftVal').innerText = `${f.sift_num_keypoints.toFixed(0)} pts`;
            document.getElementById('rowHogVal').innerText = f.hog_ortho_ratio.toFixed(4);

            document.getElementById('rowGlcmCont').innerText = isRapi ? "+14.2%" : "-8.4%";
            document.getElementById('rowEdgeCont').innerText = isRapi ? "+12.1%" : "+16.8%";
            document.getElementById('rowSiftCont').innerText = isRapi ? "+8.5%" : "+15.2%";
            document.getElementById('rowHogCont').innerText = isRapi ? "+9.3%" : "-5.1%";
        }
    </script>
</body>
</html>
"""

# ==============================================================================
# 4. FLASK SERVER ENDPOINTS
# ==============================================================================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File kosong'}), 400

    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'error': 'Format file citra tidak valid'}), 400

        # Ekstraksi 48 Fitur
        x_vec, fitur_dict = extract_48_features(img_bgr)
        x_scaled = scaler.transform(x_vec)

        # Prediksi Probabilitas
        prob_rapi = float(model.predict_proba(x_scaled)[0, 1])
        prob_berantakan = 1.0 - prob_rapi
        is_rapi = prob_rapi >= optimal_th
        confidence = (prob_rapi if is_rapi else prob_berantakan) * 100

        # Buat Thumbnail Base64 untuk preview
        h, w = img_bgr.shape[:2]
        scale = min(200 / w, 200 / h)
        img_thumb = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', img_thumb)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'is_rapi': bool(is_rapi),
            'confidence': confidence,
            'prob_rapi': prob_rapi,
            'prob_berantakan': prob_berantakan,
            'threshold': optimal_th,
            'features': fitur_dict,
            'image_b64': img_b64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# 5. PEMILIHAN PORT OTOMATIS & LAUNCHER
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
    print(" 👟 SHOESENSE AI - DRIBBBLE-STYLE MODERN DASHBOARD (UPLOAD FOTO)")
    print(f" [+] Server berjalan di: {url}")
    print(" [+] Membuka antarmuka otomatis di browser default Anda...")
    print("=" * 75 + "\n")

    # Buka browser otomatis setelah 1 detik
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False)

if __name__ == '__main__':
    run_app()
