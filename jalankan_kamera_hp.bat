@echo off
title Klasifikasi Kamera HP - Penataan Sepatu (Rapi vs Berantakan)
echo =====================================================================
echo    SISTEM KLASIFIKASI CITRA KAMERA HP (RAPI vs BERANTAKAN)
echo =====================================================================
echo.
echo [1] Memulai server web kamera HP...
echo [2] Silakan buka browser di HP Anda dan scan QR Code yang muncul di layar.
echo.
streamlit run app_kamera_hp.py --server.address 0.0.0.0 --server.port 8501
pause
