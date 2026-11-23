import React from 'react';
import { Image, FileText } from 'lucide-react';

interface HeaderProps {
  currentMode: 'image' | 'text';
  onSwitchMode: (mode: 'image' | 'text') => void;
  accuracy: number;
}

export const Header: React.FC<HeaderProps> = ({ currentMode, onSwitchMode, accuracy }) => {
  return (
    <header className="bg-white/80 backdrop-blur-xl border border-emerald-200/70 rounded-2xl shadow-emerald-soft px-4 sm:px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      {/* Kiri: Logo & Judul */}
      <div className="flex items-center gap-3.5">
        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-emerald-700 via-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-md shadow-emerald-600/25 shrink-0 text-xl">
          👟
        </div>
        <div>
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="font-display text-lg sm:text-xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-950 via-emerald-700 to-teal-600 bg-clip-text text-transparent">
              Klasifikasi Penataan Sepatu AI
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gradient-to-r from-emerald-100 to-teal-50 text-emerald-800 border border-emerald-300/70">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-600"></span>
              </span>
              Weighted Soft Voting (24 Fitur) &bull; Akurasi CV {(accuracy * 100).toFixed(2)}%
            </span>
          </div>
          <p className="text-xs sm:text-sm text-emerald-900/65 mt-0.5">
            Analisis keteraturan penataan susunan sepatu: <strong>Rapi vs Berantakan</strong>
          </p>
        </div>
      </div>

      {/* Kanan: Tab Switcher Mode */}
      <nav aria-label="Pilih Mode Klasifikasi" className="bg-emerald-100/60 p-1 rounded-xl inline-flex items-center self-start md:self-auto border border-emerald-200/80 w-full md:w-auto">
        <button
          type="button"
          onClick={() => onSwitchMode('image')}
          className={`flex-1 md:flex-none inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm transition-all duration-200 ${
            currentMode === 'image'
              ? 'bg-gradient-to-r from-emerald-600 to-teal-500 text-white font-semibold shadow-sm shadow-emerald-600/20'
              : 'text-emerald-900/75 hover:text-emerald-950 hover:bg-white/60 font-medium'
          }`}
        >
          <Image className="w-4 h-4" />
          <span>Klasifikasi Foto Sepatu</span>
        </button>
        <button
          type="button"
          onClick={() => onSwitchMode('text')}
          className={`flex-1 md:flex-none inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm transition-all duration-200 ${
            currentMode === 'text'
              ? 'bg-gradient-to-r from-emerald-600 to-teal-500 text-white font-semibold shadow-sm shadow-emerald-600/20'
              : 'text-emerald-900/75 hover:text-emerald-950 hover:bg-white/60 font-medium'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>Deskripsi Kondisi</span>
        </button>
      </nav>
    </header>
  );
};
