export interface FeatureDictionary {
  [key: string]: number;
}

export interface PredictionResponse {
  is_rapi: boolean;
  prediksi: 'rapi' | 'berantakan';
  confidence: number;
  prob_rapi: number;
  prob_berantakan: number;
  threshold: number;
  features: FeatureDictionary;
  image_b64?: string;
  inference_time_ms?: number;
  summary?: string;
}

export interface ModelInfo {
  nama_model: string;
  akurasi: number;
  roc_auc: number;
  threshold: number;
  total_fitur: number;
  fitur_cols: string[];
  kategori_fitur: {
    rgb: number;
    glcm: number;
    edge: number;
    shape: number;
    sift?: number;
  };
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  fileName: string;
  mode: 'image' | 'text';
  imageUrl?: string;
  result: PredictionResponse;
}
