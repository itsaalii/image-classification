import type { PredictionResponse, ModelInfo } from '../types';

const API_BASE = ''; // Uses relative path (handled by Vite proxy in dev or Flask in prod)

export async function classifyImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const startTime = performance.now();
  let res: Response;
  
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    // Fallback if accessed directly from external port without proxy
    res = await fetch(`http://127.0.0.1:5000/predict`, {
      method: 'POST',
      body: formData,
    });
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `Server error: ${res.statusText}`);
  }

  const data: PredictionResponse = await res.json();
  data.inference_time_ms = Math.round(performance.now() - startTime);
  if (!data.prediksi) {
    data.prediksi = data.is_rapi ? 'rapi' : 'berantakan';
  }
  return data;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/info`);
  } catch {
    res = await fetch(`http://127.0.0.1:5000/api/info`);
  }

  if (!res.ok) {
    throw new Error('Gagal memuat metadata model');
  }

  return res.json();
}
