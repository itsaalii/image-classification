import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Cpu, RefreshCw, Trash2, CheckCircle2, X, Check, XCircle, HelpCircle } from 'lucide-react';

interface UploadZoneProps {
  currentMode: 'image' | 'text';
  onFileSelected: (file: File) => void;
  onTextSubmit: (text: string) => void;
  isLoading: boolean;
  selectedFile: File | null;
  onClear: () => void;
  onTriggerClassify: () => void;
}

const TEXT_SAMPLES = {
  rapi: 'Sepatu ditata rapi sejajar berpasangan menghadap ke depan rak, tidak saling menumpuk, dan tali terselip tertib.',
  berantakan: 'Sepatu berserakan di lantai, posisinya miring acak dengan sudut diagonal, sol sepatu terbalik, dan tali sepatu kusut.',
  tanya: 'Apakah susunan sepatu ini memenuhi standar keteraturan rak sepatu ruangan tamu?',
};

export const UploadZone: React.FC<UploadZoneProps> = ({
  currentMode,
  onFileSelected,
  onTextSubmit,
  isLoading,
  selectedFile,
  onClear,
  onTriggerClassify,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [textValue, setTextValue] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [selectedFile]);

  // Handle paste image from clipboard (Ctrl+V)
  const handlePaste = useCallback((e: ClipboardEvent) => {
    if (currentMode === 'image' && e.clipboardData && e.clipboardData.items) {
      for (let i = 0; i < e.clipboardData.items.length; i++) {
        const item = e.clipboardData.items[i];
        if (item.type.indexOf('image') !== -1) {
          const file = item.getAsFile();
          if (file) {
            onFileSelected(file);
            break;
          }
        }
      }
    }
  }, [currentMode, onFileSelected]);

  useEffect(() => {
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [handlePaste]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelected(file);
      }
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      onFileSelected(file);
    }
  };

  const handleSampleClick = (type: 'rapi' | 'berantakan' | 'tanya') => {
    setTextValue(TEXT_SAMPLES[type]);
  };

  const handleClassifyClick = () => {
    if (currentMode === 'image') {
      onTriggerClassify();
    } else {
      onTextSubmit(textValue);
    }
  };

  const isButtonDisabled = isLoading || (currentMode === 'image' ? !selectedFile : textValue.trim().length === 0);

  return (
    <section className="bg-white/80 backdrop-blur-xl border border-emerald-200/70 rounded-2xl shadow-emerald-soft p-5 sm:p-6 flex flex-col justify-between gap-5 h-full">
      <div className="flex flex-col gap-4">
        {/* Header Panel Kiri */}
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="font-display text-base font-bold text-emerald-950">
              {currentMode === 'image' ? 'Input Citra Sepatu' : 'Deskripsi Penataan Sepatu'}
            </h2>
            <p className="text-xs text-emerald-900/65 mt-0.5">
              {currentMode === 'image'
                ? 'Unggah foto penataan sepatu untuk mendeteksi apakah susunan rapi atau berantakan.'
                : 'Tuliskan deskripsi kondisi penataan susunan sepatu untuk analisis keteraturan.'}
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-gradient-to-r from-emerald-100 to-lime-100/80 text-emerald-800 border border-emerald-200/80 shrink-0">
            Weighted Soft Voting (38 Fitur)
          </span>
        </div>

        {/* MODE GAMBAR */}
        {currentMode === 'image' ? (
          <div className="flex flex-col gap-4">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !previewUrl && fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-8 text-center transition-all duration-200 cursor-pointer flex flex-col items-center justify-center min-h-[300px] ${
                isDragging
                  ? 'border-emerald-500 bg-emerald-50'
                  : previewUrl
                  ? 'border-emerald-300 bg-white/90'
                  : 'border-emerald-300/90 hover:border-emerald-500 bg-gradient-to-br from-emerald-50/60 via-white/70 to-teal-50/50 hover:from-emerald-50 hover:to-teal-50/80'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp"
                className="hidden"
                onChange={handleFileInputChange}
              />

              {previewUrl ? (
                <div className="w-full flex flex-col items-center gap-3">
                  <div className="relative w-full max-w-md h-56 sm:h-64 rounded-xl overflow-hidden bg-emerald-950/5 border border-emerald-200/80 flex items-center justify-center">
                    <img
                      src={previewUrl}
                      alt="Pratinjau foto sepatu"
                      className="max-h-full max-w-full object-contain"
                    />
                    <div className="absolute top-2 right-2 flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          fileInputRef.current?.click();
                        }}
                        title="Ganti Foto"
                        className="p-2 rounded-lg bg-white/95 hover:bg-emerald-50 text-emerald-800 hover:text-emerald-700 shadow-sm border border-emerald-200/80 transition"
                      >
                        <RefreshCw className="w-4 h-4" />
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onClear();
                        }}
                        title="Hapus Foto"
                        className="p-2 rounded-lg bg-white/95 hover:bg-rose-50 text-emerald-800 hover:text-rose-600 shadow-sm border border-emerald-200/80 transition"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-emerald-900/70">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    <span className="font-semibold text-emerald-950 truncate max-w-[240px]">
                      {selectedFile?.name}
                    </span>
                    <span className="text-emerald-700/70">
                      ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-16 h-16 p-4 rounded-2xl bg-gradient-to-br from-emerald-100 to-teal-100 text-emerald-700 flex items-center justify-center border border-emerald-200/80 shadow-sm text-3xl">
                    👟
                  </div>
                  <div>
                    <p className="text-sm sm:text-base font-semibold text-emerald-950">
                      Tarik &amp; lepas foto sepatu di sini, atau{' '}
                      <span className="text-emerald-600 underline decoration-emerald-400 underline-offset-2">
                        pilih file
                      </span>
                    </p>
                    <p className="text-xs text-emerald-800/60 mt-1.5">
                      Mendukung format gambar JPG, PNG, WEBP, atau BMP (Bisa paste Ctrl+V)
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* MODE TEKS */
          <div className="flex flex-col gap-4">
            <div className="relative">
              <textarea
                rows={6}
                maxLength={1000}
                placeholder="Ketik deskripsi kondisi susunan sepatu (misal: 'Sepatu ditata rapi sejajar di rak lantai depan pintu...')"
                value={textValue}
                onChange={(e) => setTextValue(e.target.value)}
                className="w-full rounded-2xl border border-emerald-200/90 bg-emerald-50/40 focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/15 p-4 text-sm text-emerald-950 placeholder:text-emerald-800/45 outline-none transition resize-none leading-relaxed"
              />
              <div className="flex items-center justify-between mt-2 px-1">
                <button
                  type="button"
                  onClick={() => setTextValue('')}
                  className="text-xs text-emerald-800/60 hover:text-rose-600 transition inline-flex items-center gap-1 font-medium"
                >
                  <X className="w-3.5 h-3.5" /> Bersihkan teks
                </button>
                <span className="text-xs font-medium text-emerald-800/60">
                  {textValue.length} / 1000 karakter
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold text-emerald-800/75 uppercase tracking-wider">
                Atau coba deskripsi cepat:
              </span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => handleSampleClick('rapi')}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-emerald-100 to-green-100 text-emerald-800 border border-emerald-300/70 hover:from-emerald-200/80 hover:to-green-200/80 transition"
                >
                  <Check className="w-3.5 h-3.5" /> Deskripsi Rapi
                </button>
                <button
                  type="button"
                  onClick={() => handleSampleClick('berantakan')}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-rose-50 to-red-100 text-rose-900 border border-rose-200 hover:border-rose-400 hover:bg-rose-100/70 transition"
                >
                  <XCircle className="w-3.5 h-3.5 text-rose-700" /> Deskripsi Berantakan
                </button>
                <button
                  type="button"
                  onClick={() => handleSampleClick('tanya')}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-lime-50 to-emerald-50 text-emerald-900 border border-emerald-200 hover:border-emerald-400 hover:bg-emerald-100/70 transition"
                >
                  <HelpCircle className="w-3.5 h-3.5 text-teal-600" /> Pertanyaan Kerapian
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Tombol Aksi Utama */}
      <div className="pt-3 border-t border-emerald-100">
        <button
          type="button"
          disabled={isButtonDisabled}
          onClick={handleClassifyClick}
          className="w-full py-3.5 px-5 rounded-xl bg-gradient-to-r from-emerald-600 via-green-600 to-teal-500 hover:from-emerald-500 hover:via-green-500 hover:to-teal-400 active:scale-[0.99] text-white font-semibold text-sm shadow-md shadow-emerald-600/25 hover:shadow-emerald-glow transition-all duration-200 flex items-center justify-center gap-2.5 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <Cpu className="w-4 h-4" />
          <span>{isLoading ? 'Sedang Memproses...' : 'Mulai Klasifikasi Penataan'}</span>
        </button>
      </div>
    </section>
  );
};
