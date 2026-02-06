import type { ModelConfig, SafetensorsIndex } from './types';

const HF_BASE = 'https://huggingface.co';

function resolveUrl(modelId: string, path: string): string {
  return `${HF_BASE}/${modelId}/resolve/main/${path}`;
}

interface TensorMeta {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

async function fetchSafetensorsHeader(url: string): Promise<Record<string, TensorMeta>> {
  // Step 1: Fetch first 8 bytes to read header size (u64 LE)
  const sizeRes = await fetch(url, { headers: { Range: 'bytes=0-7' } });
  if (sizeRes.status !== 206) throw new Error('Range requests not supported');
  const sizeBuf = await sizeRes.arrayBuffer();
  const headerSize = Number(new DataView(sizeBuf).getBigUint64(0, true));

  // Step 2: Fetch the JSON header
  const headerRes = await fetch(url, {
    headers: { Range: `bytes=8-${8 + headerSize - 1}` },
  });
  if (headerRes.status !== 206) throw new Error('Range requests not supported');
  const header = await headerRes.json() as Record<string, TensorMeta | Record<string, unknown>>;

  // Filter out __metadata__ and return tensor entries
  const tensors: Record<string, TensorMeta> = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (meta && typeof meta === 'object' && 'shape' in meta) {
      tensors[name] = meta as TensorMeta;
    }
  }
  return tensors;
}

export async function fetchTensorParamCounts(
  modelId: string,
  weightMap: Record<string, string>,
): Promise<Map<string, number>> {
  const shardFiles = [...new Set(Object.values(weightMap))];

  const headerResults = await Promise.allSettled(
    shardFiles.map(file => fetchSafetensorsHeader(resolveUrl(modelId, file))),
  );

  const paramCounts = new Map<string, number>();
  for (const result of headerResults) {
    if (result.status !== 'fulfilled') continue;
    for (const [name, meta] of Object.entries(result.value)) {
      const count = meta.shape.reduce((a, b) => a * b, 1);
      paramCounts.set(name, count);
    }
  }

  return paramCounts;
}

export async function fetchConfig(modelId: string): Promise<ModelConfig> {
  const url = resolveUrl(modelId, 'config.json');
  const res = await fetch(url);
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(`Model "${modelId}" not found or has no config.json`);
    }
    if (res.status === 401 || res.status === 403) {
      throw new Error(`Model "${modelId}" is gated or private. Access denied.`);
    }
    throw new Error(`Failed to fetch config: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSafetensorsIndex(modelId: string): Promise<SafetensorsIndex | null> {
  // Try safetensors index first
  const safetensorsUrl = resolveUrl(modelId, 'model.safetensors.index.json');
  try {
    const res = await fetch(safetensorsUrl);
    if (res.ok) {
      return res.json();
    }
  } catch {
    // Network error, try fallback
  }

  // Try pytorch bin index as fallback
  const pytorchUrl = resolveUrl(modelId, 'pytorch_model.bin.index.json');
  try {
    const res = await fetch(pytorchUrl);
    if (res.ok) {
      return res.json();
    }
  } catch {
    // Network error
  }

  return null;
}
