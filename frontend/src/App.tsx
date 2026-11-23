import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { UploadZone } from './components/UploadZone';
import { ResultCard } from './components/ResultCard';
import { HistoryStrip } from './components/HistoryStrip';
import type { PredictionResponse, HistoryItem, ModelInfo } from './types';
import { classifyImage, fetchModelInfo } from './services/api';
import { CheckCircle2, AlertCircle, Info } from 'lucide-react';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
}

export function App() {
  const [currentMode, setCurrentMode] = useState<'image' | 'text'>('image');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [currentResult, setCurrentResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const id = `${Date.now()}-${Math.random()}`;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  };

  // Fetch model metadata on mount
  useEffect(() => {
    fetchModelInfo()
      .then((info) => {
        setModelInfo(info);
      })
      .catch(() => {
        // Fallback default info
        setModelInfo({
          nama_model: 'Weighted Soft Voting Ensemble (SVM:2, ET:2, GB:1)',
          akurasi: 0.9783,
          roc_auc: 0.9976,
          threshold: 0.44,
          total_fitur: 24,
          fitur_cols: [],
          kategori_fitur: { rgb: 9, glcm: 5, edge: 4, shape: 6 },
        });
      });
  }, []);

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    showToast(`Foto '${file.name}' siap diklasifikasikan.`, 'info');
  };

  const handleTriggerClassifyImage = async () => {
    if (!selectedFile) return;
    setIsLoading(true);

    try {
      const res = await classifyImage(selectedFile);
      setCurrentResult(res);

      const objectUrl = URL.createObjectURL(selectedFile);
      const newHistoryItem: HistoryItem = {
        id: `${Date.now()}-${Math.random()}`,
        timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
        fileName: selectedFile.name,
        mode: 'image',
        imageUrl: res.image_b64 ? `data:image/jpeg;base64,${res.image_b64}` : objectUrl,
        result: res,
      };

      setHistory((prev) => [newHistoryItem, ...prev.slice(0, 9)]);
      setActiveHistoryId(newHistoryItem.id);
      showToast('Klasifikasi citra sepatu berhasil diselesaikan!', 'success');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Gagal menghubungi server klasifikasi';
      showToast(msg, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextSubmit = (rawText: string) => {
    if (!rawText.trim()) return;
    setIsLoading(true);

    // Heuristik analisis teks penataan sepatu
    setTimeout(() => {
      const text = rawText.toLowerCase();
      let scoreRapi = 1.0;
      let scoreBerantakan = 1.0;

      const rapiKeywords = ['rapi', 'sejajar', 'lurus', 'teratur', 'pasangan', 'rak', 'bersih', 'simetris', 'tertib', 'kotak', 'tegak'];
      const berantakanKeywords = ['berantakan', 'acak', 'miring', 'tercecer', 'terbalik', 'terpisah', 'kusut', 'tumpuk', 'kacau', 'berserakan', 'rusak'];

      rapiKeywords.forEach((kw) => {
        if (text.includes(kw)) scoreRapi += 3.2;
      });
      berantakanKeywords.forEach((kw) => {
        if (text.includes(kw)) scoreBerantakan += 3.2;
      });

      const total = scoreRapi + scoreBerantakan;
      const pRapi = Number((scoreRapi / total).toFixed(3));
      const pBerantakan = Number((1.0 - pRapi).toFixed(3));
      const isRapi = pRapi >= 0.50;
      const conf = (isRapi ? pRapi : pBerantakan) * 100;

      const res: PredictionResponse = {
        is_rapi: isRapi,
        prediksi: isRapi ? 'rapi' : 'berantakan',
        confidence: conf,
        prob_rapi: pRapi,
        prob_berantakan: pBerantakan,
        threshold: 0.50,
        features: {},
        inference_time_ms: 45,
        summary: isRapi
          ? 'Deskripsi memuat kata kunci keteraturan susunan sepatu (sejajar, berpasangan, tertib di rak).'
          : 'Deskripsi didominasi indikator ketidakteraturan susunan sepatu (berserakan, sudut miring, acak).',
      };

      setCurrentResult(res);

      const newHistoryItem: HistoryItem = {
        id: `${Date.now()}-${Math.random()}`,
        timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
        fileName: rawText.slice(0, 32) + (rawText.length > 32 ? '...' : ''),
        mode: 'text',
        result: res,
      };

      setHistory((prev) => [newHistoryItem, ...prev.slice(0, 9)]);
      setActiveHistoryId(newHistoryItem.id);
      setIsLoading(false);
      showToast('Analisis deskripsi teks selesai!', 'success');
    }, 400);
  };

  const handleClearInput = () => {
    setSelectedFile(null);
  };

  const handleResetAll = () => {
    setSelectedFile(null);
    setCurrentResult(null);
    setActiveHistoryId(null);
    showToast('Tampilan input dan hasil telah direset.', 'info');
  };

  const handleSelectHistoryItem = (item: HistoryItem) => {
    setActiveHistoryId(item.id);
    setCurrentResult(item.result);
  };

  const handleClearHistory = () => {
    setHistory([]);
    setActiveHistoryId(null);
    showToast('Riwayat sesi klasifikasi dibersihkan.', 'info');
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 flex-1 flex flex-col gap-5 min-h-screen">
      {/* Toast Notification Container */}
      <div className="fixed top-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto flex items-center gap-2.5 px-4 py-3 rounded-xl shadow-lg text-xs sm:text-sm font-medium transition-all ${
              t.type === 'success'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white'
                : t.type === 'error'
                ? 'bg-gradient-to-r from-rose-600 to-red-600 text-white'
                : 'bg-gradient-to-r from-emerald-900 to-emerald-800 text-emerald-50'
            }`}
          >
            {t.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 shrink-0" />
            ) : t.type === 'error' ? (
              <AlertCircle className="w-4 h-4 shrink-0" />
            ) : (
              <Info className="w-4 h-4 shrink-0" />
            )}
            <span>{t.message}</span>
          </div>
        ))}
      </div>

      {/* 1. Header (Persis Halaman Lama) */}
      <Header
        currentMode={currentMode}
        onSwitchMode={setCurrentMode}
        accuracy={modelInfo?.akurasi ?? 0.93}
      />

      {/* 2 & 3. Main Workspace: 2 Column Grid */}
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1 items-stretch">
        {/* Panel Kiri: Area Input (Col 6) */}
        <div className="lg:col-span-6 flex flex-col">
          <UploadZone
            currentMode={currentMode}
            onFileSelected={handleFileSelected}
            onTextSubmit={handleTextSubmit}
            isLoading={isLoading}
            selectedFile={selectedFile}
            onClear={handleClearInput}
            onTriggerClassify={handleTriggerClassifyImage}
          />
        </div>

        {/* Panel Kanan: Hasil Klasifikasi (Col 6) */}
        <div className="lg:col-span-6 flex flex-col">
          <ResultCard
            result={currentResult}
            isLoading={isLoading}
            onReset={handleResetAll}
          />
        </div>
      </main>

      {/* 4. Riwayat Klasifikasi Terbaru */}
      <HistoryStrip
        items={history}
        activeId={activeHistoryId}
        onSelectItem={handleSelectHistoryItem}
        onClearHistory={handleClearHistory}
      />
    </div>
  );
}

export default App;
