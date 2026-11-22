@echo off
title ShoeSense AI - React 19 + Vite + Tailwind Dashboard
color 0a
cls
echo ===============================================================================
echo        SHOESENSE AI - KLASIFIKASI PENATAAN SEPATU (REACT 19 + TAILWIND)
echo        Weighted Soft Voting (SVM:2, ET:2, GB:1) - 24 Fitur Multimodal
echo ===============================================================================
echo.
echo  [+] Memulai backend Flask dan melayani UI React 19 di http://127.0.0.1:5000...
echo.

py -3.13 app_upload_modern.py || python app_upload_modern.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Terjadi kesalahan saat menjalankan backend.
    pause
)
