import React from 'react';
import type { HistoryItem } from '../types';
import { History } from 'lucide-react';

interface HistoryStripProps {
  items: HistoryItem[];
  activeId: string | null;
  onSelectItem: (item: HistoryItem) => void;
  onClearHistory: () => void;
}

export const HistoryStrip: React.FC<HistoryStripProps> = ({
  items,
  activeId,
  onSelectItem,
  onClearHistory,
}) => {
  return (
    <section className="bg-white/80 backdrop-blur-xl border border-emerald-200/70 rounded-2xl shadow-emerald-soft px-5 py-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-100 text-emerald-700 flex items-center justify-center">
            <History className="w-4 h-4" />
          </div>
          <h2 className="font-display text-sm font-bold text-emerald-950">
            Riwayat Klasifikasi Penataan Sepatu
          </h2>
          <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-800 border border-emerald-200/70">
            {items.length}
          </span>
        </div>

        {items.length > 0 && (
          <button
            type="button"
            onClick={onClearHistory}
            className="text-xs font-semibold text-emerald-800/60 hover:text-rose-600 transition"
          >
            Hapus Riwayat
          </button>
        )}
      </div>

      <div className="flex items-center gap-3 overflow-x-auto pb-2 pt-0.5">
        {items.length === 0 ? (
          <p className="text-xs text-emerald-800/60 py-2">
            Belum ada riwayat pada sesi ini. Lakukan klasifikasi penataan pertama Anda di atas.
          </p>
        ) : (
          items.map((item) => {
            const isActive = item.id === activeId;
            const isRapi = item.result.is_rapi;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectItem(item)}
                className={`shrink-0 w-64 text-left p-3 rounded-xl border transition-all duration-150 flex items-center gap-3 cursor-pointer ${
                  isActive
                    ? 'border-emerald-500 bg-gradient-to-br from-emerald-50 to-teal-50/70 ring-2 ring-emerald-500/20 shadow-sm'
                    : 'border-emerald-200/80 hover:border-emerald-400 hover:bg-emerald-50/40 bg-white/90'
                }`}
              >
                {item.imageUrl ? (
                  <img
                    src={item.imageUrl}
                    alt="Thumb"
                    className="w-10 h-10 rounded-lg object-cover border border-emerald-200 shrink-0"
                  />
                ) : (
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-100 to-teal-100 text-emerald-700 flex items-center justify-center shrink-0 border border-emerald-200/70 text-lg">
                    👟
                  </div>
                )}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1.5">
                    <span className="text-[10px] font-medium text-emerald-800/60">
                      {item.timestamp} &bull; {item.mode === 'image' ? 'Citra' : 'Teks'}
                    </span>
                    <span className="text-[11px] font-bold text-emerald-700">
                      {item.result.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs font-semibold text-emerald-950 truncate mt-0.5">
                    {item.fileName}
                  </p>
                  <span
                    className={`inline-block mt-1 px-2 py-0.5 rounded-md text-[10px] font-semibold ${
                      isRapi
                        ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                        : 'bg-rose-100 text-rose-800 border border-rose-300'
                    }`}
                  >
                    {isRapi ? 'Penataan Rapi' : 'Penataan Berantakan'}
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </section>
  );
};
