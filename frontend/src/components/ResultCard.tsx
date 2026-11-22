import React, { useState } from 'react';
import type { PredictionResponse } from '../types';
import { Copy, RotateCcw, BrainCircuit, Zap, Layers, Info, Check, CheckCircle2, AlertTriangle } from 'lucide-react';

interface ResultCardProps {
  result: PredictionResponse | null;
  isLoading: boolean;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, isLoading, onReset }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!result) return;
    const isRapi = result.is_rapi;
    const pRapi = (result.prob_rapi * 100).toFixed(1);
    const pBerantakan = (result.prob_berantakan * 100).toFixed(1);

    const text = `Hasil Klasifikasi Penataan Sepatu AI
Status: ${isRapi ? 'Penataan Rapi' : 'Penataan Berantakan'} (${result.confidence.toFixed(1)}%)
Waktu Inferensi: ${result.inference_time_ms || 0} ms
Model: Weighted Soft Voting (SVM:2, ET:2, GB:1)
Distribusi:
- Peluang Rapi: ${pRapi}%
- Peluang Berantakan: ${pBerantakan}%
Ambang Batas Optimal: ${result.threshold.toFixed(2)}
Ringkasan: ${result.summary || (isRapi ? 'Susunan sepatu rapi berpasangan sejajar dengan orientasi konsisten.' : 'Sepatu berserakan dengan orientasi acak miring atau tali tidak beraturan.')}`;

    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isRapi = result?.is_rapi ?? true;
  const pRapi = (result?.prob_rapi ? result.prob_rapi * 100 : 0);
  const pBerantakan = (result?.prob_berantakan ? result.prob_berantakan * 100 : 0);

  return (
    <section className="bg-white/80 backdrop-blur-xl border border-emerald-200/70 rounded-2xl shadow-emerald-soft p-5 sm:p-6 flex flex-col justify-between h-full min-h-[460px]">
      <div>
        {/* Header Panel Kanan */}
        <div className="flex items-center justify-between gap-2 mb-4">
          <div>
            <h2 className="font-display text-base font-bold text-emerald-950">
              Hasil Klasifikasi Penataan
            </h2>
            <p className="text-xs text-emerald-900/65 mt-0.5">
              Prediksi status: <strong>Penataan Rapi</strong> vs <strong>Penataan Berantakan</strong>.
            </p>
          </div>

          {result && !isLoading && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200/80 transition"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Tersalin' : 'Salin Hasil'}</span>
              </button>
              <button
                type="button"
                onClick={onReset}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/80 hover:bg-rose-50 text-emerald-900/75 hover:text-rose-600 border border-emerald-200/80 hover:border-rose-200 transition"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            </div>
          )}
        </div>

        {/* STATE 1: KOSONG (AWAL) */}
        {!result && !isLoading && (
          <div className="flex flex-col items-center justify-center text-center py-12 px-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-100/90 to-teal-100/80 text-emerald-600 flex items-center justify-center mb-4 border border-emerald-200/80 shadow-sm text-3xl">
              👟
            </div>
            <h3 className="text-sm font-bold text-emerald-950">Belum Ada Analisis Penataan</h3>
            <p className="text-xs sm:text-sm text-emerald-900/65 max-w-xs mt-1.5 leading-relaxed">
              Unggah foto sepatu di sebelah kiri untuk melihat apakah susunan terdeteksi <strong>Rapi</strong> atau <strong>Berantakan</strong> secara instan.
            </p>
          </div>
        )}

        {/* STATE 2: MEMPROSES / LOADING */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center text-center py-12 px-4">
            <div className="relative w-16 h-16 flex items-center justify-center mb-4">
              <div className="absolute inset-0 rounded-2xl border-4 border-emerald-100"></div>
              <div className="absolute inset-0 rounded-2xl border-4 border-emerald-600 border-t-lime-400 animate-spin"></div>
              <BrainCircuit className="w-7 h-7 text-emerald-600 animate-pulse" />
            </div>
            <h3 className="text-sm font-bold text-emerald-950">Menganalisis 38 Fitur Multimodal...</h3>
            <p className="text-xs text-emerald-800/70 mt-1">
              Mengekstrak GLCM CLAHE, Tepi Canny, dan SIFT BoVW
            </p>
            <div className="w-52 h-2 bg-emerald-100 rounded-full overflow-hidden mt-4 border border-emerald-200/70">
              <div className="h-full bg-gradient-to-r from-emerald-600 via-green-500 to-lime-400 rounded-full animate-pulse w-3/4"></div>
            </div>
          </div>
        )}

        {/* STATE 3: HASIL PREDIKSI */}
        {result && !isLoading && (
          <div className="flex flex-col gap-4">
            {/* Kartu Prediksi Utama */}
            <div
              className={`rounded-2xl border p-4 sm:p-5 shadow-sm transition-all duration-300 ${
                isRapi
                  ? 'bg-gradient-to-br from-emerald-50 via-teal-50/60 to-white border-emerald-300'
                  : 'bg-gradient-to-br from-rose-50 via-red-50/50 to-white border-rose-300'
              }`}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap items-center gap-2 mb-1.5">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-800/70">
                      Status Klasifikasi
                    </span>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold text-white shadow-sm ${
                        isRapi
                          ? 'bg-gradient-to-r from-emerald-600 to-teal-600'
                          : 'bg-gradient-to-r from-rose-600 to-red-600'
                      }`}
                    >
                      {result.confidence >= 80 ? 'Keyakinan Tinggi' : 'Keyakinan Cukup'}
                    </span>
                  </div>
                  <h3
                    className={`font-display text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2 ${
                      isRapi ? 'text-emerald-800' : 'text-rose-800'
                    }`}
                  >
                    {isRapi ? (
                      <CheckCircle2 className="w-7 h-7 text-emerald-600 shrink-0" />
                    ) : (
                      <AlertTriangle className="w-7 h-7 text-rose-600 shrink-0" />
                    )}
                    <span>{isRapi ? 'Penataan Rapi' : 'Penataan Berantakan'}</span>
                  </h3>
                </div>

                {/* Skor Kepercayaan */}
                <div className="text-left sm:text-right">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-800/70 block">
                    Tingkat Keyakinan
                  </span>
                  <span
                    className={`font-display text-2xl sm:text-3xl font-extrabold ${
                      isRapi ? 'text-emerald-700' : 'text-rose-700'
                    }`}
                  >
                    {result.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Metadata Inferensi */}
              <div className="mt-3.5 pt-3 border-t border-emerald-200/70 flex flex-wrap items-center justify-between gap-2 text-xs text-emerald-900/70">
                <span className="inline-flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-emerald-600" />
                  Waktu inferensi: <strong className="text-emerald-950">{result.inference_time_ms || 0} ms</strong>
                </span>
                <span className="inline-flex items-center gap-1.5 text-emerald-900/75 font-medium">
                  <Layers className="w-3.5 h-3.5 text-teal-600" />
                  <span>Weighted Soft Voting (SVM:2, ET:2, GB:1)</span>
                </span>
              </div>
            </div>

            {/* Distribusi Probabilitas */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-emerald-800/75">
                  Distribusi Probabilitas Kelas
                </h4>
                <span className="text-[11px] text-emerald-800/60 font-medium">
                  Ambang Batas: <strong>{(result.threshold * 100).toFixed(0)}%</strong>
                </span>
              </div>

              <div className="flex flex-col gap-2.5">
                {/* Bar Rapi */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-xs font-semibold text-emerald-950">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                      Penataan Rapi
                    </span>
                    <span>{pRapi.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 bg-emerald-100 rounded-full overflow-hidden border border-emerald-200/80">
                    <div
                      style={{ width: `${pRapi}%` }}
                      className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all duration-700"
                    ></div>
                  </div>
                </div>

                {/* Bar Berantakan */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-xs font-semibold text-emerald-950">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                      Penataan Berantakan
                    </span>
                    <span>{pBerantakan.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 bg-rose-100 rounded-full overflow-hidden border border-rose-200/80">
                    <div
                      style={{ width: `${pBerantakan}%` }}
                      className="h-full bg-gradient-to-r from-rose-500 to-red-500 rounded-full transition-all duration-700"
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Ringkasan Analisis Karakteristik */}
            <div className="rounded-xl bg-emerald-50/70 border border-emerald-200/70 p-3.5 flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-600 to-teal-500 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                <Info className="w-4 h-4" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-emerald-950">
                  Ringkasan Analisis Karakteristik Citra
                </h4>
                <p className="text-xs text-emerald-900/80 mt-1 leading-relaxed">
                  {result.summary ||
                    (isRapi
                      ? 'Sepatu berpasangan ditata sejajar berdampingan, sudut orientasi konsisten, tidak saling menumpuk, dan tali sepatu terselip tertib.'
                      : 'Sepatu berserakan dengan orientasi acak miring, letak pasangan terpisah, posisi miring menampakkan sol, atau tali sepatu kusut.')}
                </p>
              </div>
            </div>

            {/* Nilai Fitur Kunci */}
            {result.features && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px]">
                <div className="p-2.5 rounded-xl bg-emerald-50/50 border border-emerald-200/60">
                  <span className="text-emerald-800/60 block">GLCM Entropy</span>
                  <strong className="text-emerald-950 font-bold">
                    {result.features['glcm_entropy']?.toFixed(2) ?? '-'}
                  </strong>
                </div>
                <div className="p-2.5 rounded-xl bg-emerald-50/50 border border-emerald-200/60">
                  <span className="text-emerald-800/60 block">Edge Density</span>
                  <strong className="text-emerald-950 font-bold">
                    {result.features['edge_density']?.toFixed(3) ?? '-'}
                  </strong>
                </div>
                <div className="p-2.5 rounded-xl bg-emerald-50/50 border border-emerald-200/60">
                  <span className="text-emerald-800/60 block">SIFT Keypoints</span>
                  <strong className="text-emerald-950 font-bold">
                    {result.features['sift_num_keypoints']?.toFixed(0) ?? '-'} pts
                  </strong>
                </div>
                <div className="p-2.5 rounded-xl bg-emerald-50/50 border border-emerald-200/60">
                  <span className="text-emerald-800/60 block">Contour Solidity</span>
                  <strong className="text-emerald-950 font-bold">
                    {result.features['contour_solidity']?.toFixed(2) ?? '-'}
                  </strong>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
};
