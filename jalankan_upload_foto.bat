@echo off
title Klasifikasi Penataan Sepatu AI (Rapi vs Berantakan)
color 0a
cls
echo ===============================================================================
echo        KLASIFIKASI PENATAAN SEPATU AI (RAPI VS BERANTAKAN)
echo        Weighted Soft Voting (SVM:2, ET:2, GB:1) - 38 Fitur Multimodal
echo ===============================================================================
echo.
echo  Membuka antarmuka web modern di browser Anda...
echo.

start "" "%~dp0klasifikasi_ai.html"

echo [OK] Antarmuka telah dibuka di browser default Anda (Chrome / Edge).
echo.
echo Jika ingin menjalankan via Streamlit, ketik: streamlit run web_upload_foto.py
echo.
timeout /t 4 >nul
exit

