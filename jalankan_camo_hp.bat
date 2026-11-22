@echo off
title Klasifikasi Camo Studio (Kamera HP) - Weighted Soft Voting (SVM:2, ET:2, GB:1)
color 0A
echo ==============================================================================
echo    KLASIFIKASI CITRA PENATAAN SEPATU DENGAN KAMERA HP (CAMO STUDIO)
echo    METODE: WEIGHTED SOFT VOTING (SVM:2, ET:2, GB:1) - 24 FITUR (AKURASI 97.83%%)
echo ==============================================================================
echo.
echo Pastikan aplikasi Camo Studio di laptop ^& HP Anda sudah aktif dan terhubung.
echo (Tips: Tekan [TAB] pada jendela kamera untuk beralih antara Camo Studio dan Webcam)
echo.
echo Menjalankan aplikasi kamera Camo Studio...
py -3.13 camo_kamera_hp.py %* || python camo_kamera_hp.py %*
pause
