import os
import cv2
import time
import json
import io
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from concurrent.futures import ProcessPoolExecutor

folder_dataset = 'foto-psd'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Precompute index arrays for GLCM 256x256
i_idx, j_idx = np.indices((256, 256))
diff_abs = 1.0 + np.abs(i_idx - j_idx)
diff_sq = (i_idx - j_idx)**2

def process_image(args):
    file_path, file, kelas = args
    try:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            img_bgr = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        
        # 1. Fitur Warna (RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mean_r, mean_g, mean_b = img_rgb.mean(axis=(0, 1))
        std_r, std_g, std_b = img_rgb.std(axis=(0, 1))
        
        # Hitung skewness per kanal agar hemat memori (bebas OOM untuk foto resolusi tinggi)
        sk_channels = []
        for c in range(3):
            ch = img_rgb[:, :, c].astype(np.float64)
            sk_val = float(np.mean(((ch - [mean_r, mean_g, mean_b][c]) / [std_r, std_g, std_b][c])**3))
            sk_channels.append(sk_val)
        skew_r, skew_g, skew_b = sk_channels
        
        # 2. Fitur Tekstur GLCM (Slide 19 & 20)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Menghitung GLCM ternormalisasi (jarak d=1, sudut theta=0 derajat)
        glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
        p = glcm[:, :, 0, 0]
        
        # Slide 19: Kontras (Pers. 6)
        glcm_contrast = float(np.sum(diff_sq * p))
        
        # Slide 19: Entropi (Pers. 5)
        mask = p > 0
        glcm_entropy = float(-np.sum(p[mask] * np.log2(p[mask])))
        
        # Slide 20: Korelasi (Pers. 7)
        glcm_correlation = float(graycoprops(glcm, 'correlation')[0, 0])
        
        # Slide 20: Energi / ASM (Pers. 8)
        glcm_energy = float(np.sum(p**2))
        
        # Slide 20: Homogenitas (Pers. 9)
        glcm_homogeneity = float(np.sum(p / diff_abs))
        
        return {
            'nama_file': file,
            # Fitur Warna (RGB)
            'mean_r': float(mean_r),
            'mean_g': float(mean_g),
            'mean_b': float(mean_b),
            'std_r': float(std_r),
            'std_g': float(std_g),
            'std_b': float(std_b),
            'skew_r': float(skew_r),
            'skew_g': float(skew_g),
            'skew_b': float(skew_b),
            # Fitur Tekstur GLCM (Slide 19 & 20)
            'glcm_contrast': glcm_contrast,
            'glcm_entropy': glcm_entropy,
            'glcm_correlation': glcm_correlation,
            'glcm_energy': glcm_energy,
            'glcm_homogeneity': glcm_homogeneity,
            # Label Kelas
            'kelas': kelas
        }
    except Exception as e:
        print(f"Error pada {file}: {e}")
        return None

def main():
    t0 = time.time()
    tasks = []
    for root, dirs, files in os.walk(folder_dataset):
        for file in files:
            if file.lower().endswith(valid_extensions):
                tasks.append((os.path.join(root, file), file, os.path.basename(root)))
                
    print(f"Total file yang akan diproses: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(process_image, tasks))
        
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df = df.sort_values(by=['kelas', 'nama_file']).reset_index(drop=True)
    
    print(f"Ekstraksi selesai dalam {time.time()-t0:.2f} detik.")
    print("Total baris berhasil:", len(df))
    print("Kolom hasil ekstraksi:", df.columns.tolist())
    print("Distribusi Kelas:")
    print(df['kelas'].value_counts())
    
    # 1. Simpan CSV
    df.to_csv('hasil_ekstraksi_fitur.csv', index=False)
    df.to_csv('hasil_ekstraksi_fitur_terbaru.csv', index=False)
    print("Berhasil menyimpan ke hasil_ekstraksi_fitur.csv dan hasil_ekstraksi_fitur_terbaru.csv")
    
    # 2. Update main.ipynb
    with open('main.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Cell 0: Markdown
    nb['cells'][0]['source'] = [
        '# Proyek Sains Data - Ekstraksi Fitur Citra\n',
        '\n',
        'Notebook ini melakukan ekstraksi fitur warna (RGB) dan fitur tekstur GLCM dari dataset citra pada folder `foto-psd`:\n',
        '- **Fitur Warna (RGB - Slide 9-11)**:\n',
        '  - Rata-rata (Mean): `mean_r`, `mean_g`, `mean_b`\n',
        '  - Standar Deviasi (Standard Deviation): `std_r`, `std_g`, `std_b`\n',
        '  - Kemiringan Distribusi (Skewness): `skew_r`, `skew_g`, `skew_b`\n',
        '- **Fitur Tekstur GLCM (Slide 19 & 20)**:\n',
        '  - Kontras (Contrast - Pers. 6): `glcm_contrast`\n',
        '  - Entropi (Entropy - Pers. 5): `glcm_entropy`\n',
        '  - Korelasi (Correlation - Pers. 7): `glcm_correlation`\n',
        '  - Energi (Energy / ASM - Pers. 8): `glcm_energy`\n',
        '  - Homogenitas (Homogeneity - Pers. 9): `glcm_homogeneity`\n',
        '- **Label / Target**:\n',
        '  - `kelas`: Diambil otomatis dari nama subfolder (`rapi`, `berantakan`)\n'
    ]
    
    # Cell 1: Code
    cell1_code = [
        'import os\n',
        'import cv2\n',
        'import numpy as np\n',
        'import pandas as pd\n',
        'from scipy.stats import skew\n',
        'from skimage.feature import graycomatrix, graycoprops\n',
        '\n',
        '# 1. Path folder dataset citra terbaru\n',
        "folder_dataset = 'foto-psd'\n",
        '\n',
        '# Format file citra yang didukung\n',
        "valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')\n",
        '\n',
        '# List untuk menampung hasil ekstraksi fitur\n',
        'data_fitur = []\n',
        '\n',
        '# Precompute matriks indeks untuk perhitungan GLCM (Slide 19 & 20)\n',
        'i_idx, j_idx = np.indices((256, 256))\n',
        'diff_abs = 1.0 + np.abs(i_idx - j_idx)\n',
        'diff_sq = (i_idx - j_idx)**2\n',
        '\n',
        '# 2. Iterasi setiap sub-folder (kelas) dan file gambar di dalam folder foto-psd\n',
        'for root, dirs, files in os.walk(folder_dataset):\n',
        '    for file in files:\n',
        '        if file.lower().endswith(valid_extensions):\n',
        '            file_path = os.path.join(root, file)\n',
        '            \n',
        "            # Kelas didapat dari nama folder tempat citra berada (misal: 'rapi', 'berantakan')\n",
        '            kelas = os.path.basename(root)\n',
        '            \n',
        '            # Membaca citra menggunakan OpenCV (format bawaan: BGR)\n',
        '            img_bgr = cv2.imread(file_path)\n',
        '            \n',
        '            # Fallback jika cv2.imread mengembalikan None (misal handling path tertentu)\n',
        '            if img_bgr is None:\n',
        '                img_bgr = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)\n',
        '                \n',
        '            if img_bgr is not None:\n',
        '                # -----------------------------------------------------------------\n',
        '                # A. FITUR WARNA (RGB) - Slide 9, 10, 11\n',
        '                # -----------------------------------------------------------------\n',
        '                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n',
        '                \n',
        '                # Mean (Rata-rata) tiap kanal warna\n',
        '                mean_r, mean_g, mean_b = img_rgb.mean(axis=(0, 1))\n',
        '                \n',
        '                # Standar Deviasi tiap kanal warna\n',
        '                std_r, std_g, std_b = img_rgb.std(axis=(0, 1))\n',
        '                \n',
        '                # Skewness (Kemencengan distribusi) tiap kanal warna\n',
        '                skew_r, skew_g, skew_b = skew(img_rgb.reshape(-1, 3), axis=0)\n',
        '                \n',
        '                # -----------------------------------------------------------------\n',
        '                # B. FITUR TEKSTUR GLCM - Slide 19 & 20\n',
        '                # -----------------------------------------------------------------\n',
        '                # Konversi ke Grayscale 8-bit untuk analisis spasial tekstur\n',
        '                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)\n',
        '                \n',
        '                # Menghitung matriks GLCM ternormalisasi (jarak d=1, sudut theta=0 derajat)\n',
        '                glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)\n',
        '                p = glcm[:, :, 0, 0]\n',
        '                \n',
        '                # 1. Kontras (Slide 19, Pers. 6): sum (i - j)^2 * P(i, j)\n',
        '                glcm_contrast = float(np.sum(diff_sq * p))\n',
        '                \n',
        '                # 2. Entropi (Slide 19, Pers. 5): -sum P(i, j) * log2(P(i, j))\n',
        '                mask = p > 0\n',
        '                glcm_entropy = float(-np.sum(p[mask] * np.log2(p[mask])))\n',
        '                \n',
        '                # 3. Korelasi (Slide 20, Pers. 7)\n',
        '                glcm_correlation = float(graycoprops(glcm, \'correlation\')[0, 0])\n',
        '                \n',
        '                # 4. Energi / ASM (Slide 20, Pers. 8): sum P(i, j)^2\n',
        '                glcm_energy = float(np.sum(p**2))\n',
        '                \n',
        '                # 5. Homogenitas (Slide 20, Pers. 9): sum P(i, j) / (1 + |i - j|)\n',
        '                glcm_homogeneity = float(np.sum(p / diff_abs))\n',
        '                \n',
        '                # -----------------------------------------------------------------\n',
        '                # Simpan seluruh fitur ke list\n',
        '                # -----------------------------------------------------------------\n',
        '                data_fitur.append({\n',
        "                    'nama_file': file,\n",
        "                    'mean_r': float(mean_r),\n",
        "                    'mean_g': float(mean_g),\n",
        "                    'mean_b': float(mean_b),\n",
        "                    'std_r': float(std_r),\n",
        "                    'std_g': float(std_g),\n",
        "                    'std_b': float(std_b),\n",
        "                    'skew_r': float(skew_r),\n",
        "                    'skew_g': float(skew_g),\n",
        "                    'skew_b': float(skew_b),\n",
        "                    'glcm_contrast': glcm_contrast,\n",
        "                    'glcm_entropy': glcm_entropy,\n",
        "                    'glcm_correlation': glcm_correlation,\n",
        "                    'glcm_energy': glcm_energy,\n",
        "                    'glcm_homogeneity': glcm_homogeneity,\n",
        "                    'kelas': kelas\n",
        '                })\n',
        '\n',
        '# 3. Membuat DataFrame dari hasil ekstraksi\n',
        'df = pd.DataFrame(data_fitur)\n',
        '\n',
        'print(f"Total citra yang berhasil diekstraksi: {len(df)}")\n',
        'df.head()\n'
    ]
    nb['cells'][1]['source'] = cell1_code
    nb['cells'][1]['execution_count'] = 1
    
    # Format Cell 1 outputs
    head_df = df.head()
    head_html = head_df.to_html()
    head_text = head_df.to_string()
    nb['cells'][1]['outputs'] = [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [f'Total citra yang berhasil diekstraksi: {len(df)}\n']
        },
        {
            'data': {
                'text/html': [head_html + '\n'],
                'text/plain': [head_text + '\n']
            },
            'execution_count': 1,
            'metadata': {},
            'output_type': 'execute_result'
        }
    ]
    
    # Format Cell 2 outputs
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    vc_str = df['kelas'].value_counts().to_string()
    cell2_stdout = (
        'Informasi Dataset:\n' +
        info_str +
        '\nJumlah Sampel per Kelas:\n' +
        vc_str + '\nName: count, dtype: int64\n'
    )
    nb['cells'][2]['execution_count'] = 2
    nb['cells'][2]['outputs'] = [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': [s + '\n' for s in cell2_stdout.splitlines()]
        }
    ]
    
    # Format Cell 3 outputs
    nb['cells'][3]['execution_count'] = 3
    nb['cells'][3]['outputs'] = [
        {
            'name': 'stdout',
            'output_type': 'stream',
            'text': ["Hasil ekstraksi fitur berhasil disimpan ke 'hasil_ekstraksi_fitur.csv'\n"]
        }
    ]
    
    with open('main.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook main.ipynb berhasil diperbarui!")

if __name__ == '__main__':
    main()
