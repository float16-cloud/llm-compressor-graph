import type { ModelConfig } from './types';

export interface KVCacheEstimate {
  seqLen: number;
  label: string;
  fp16Bytes: number;
  fp8Bytes: number;
  /** Fixed-size state from linear/SSM layers (bytes, FP16). 0 if pure attention. */
  linearStateFp16: number;
}

/**
 * Estimate KV cache memory for given context lengths.
 *
 * For pure attention models:
 *   2 (K+V) × num_layers × num_kv_heads × head_dim × seq_len × bytes
 *
 * For hybrid models (full_attention_interval > 0):
 *   Only full-attention layers contribute growing KV cache.
 *   Linear/DeltaNet layers contribute a fixed-size recurrent state:
 *     per layer = num_key_heads × key_dim × value_dim (the S = Σ kᵢ⊗vᵢ matrix)
 */
export function estimateKVCache(config: ModelConfig): KVCacheEstimate[] | null {
  const totalLayers = config.num_hidden_layers ?? config.text_config?.num_hidden_layers;
  const H = config.hidden_size ?? config.text_config?.hidden_size;
  const N = config.num_attention_heads ?? config.text_config?.num_attention_heads;

  if (!totalLayers || !H || !N) return null;

  const KV = config.num_key_value_heads ?? config.text_config?.num_key_value_heads ?? N;
  const headDim = config.head_dim ?? config.text_config?.head_dim ?? Math.floor(H / N);

  // Hybrid architecture: only every Nth layer uses full attention
  const interval = config.full_attention_interval;
  const attnLayers = interval && interval > 1
    ? Math.floor(totalLayers / interval)
    : totalLayers;
  const linearLayers = totalLayers - attnLayers;

  // Fixed-size recurrent state for linear/DeltaNet layers
  // State per layer = num_heads × d_key × d_value (the outer-product accumulator)
  let linearStateFp16 = 0;
  if (linearLayers > 0 && config.linear_num_key_heads && config.linear_key_head_dim && config.linear_value_head_dim) {
    const statePerLayer = config.linear_num_key_heads * config.linear_key_head_dim * config.linear_value_head_dim;
    linearStateFp16 = linearLayers * statePerLayer * 2; // ×2 for FP16 bytes
  }

  const SEQ_LENGTHS = [
    { seqLen: 32_768, label: '32k' },
    { seqLen: 65_536, label: '64k' },
    { seqLen: 131_072, label: '128k' },
  ];

  return SEQ_LENGTHS.map(({ seqLen, label }) => {
    // KV cache from full attention layers (grows with seq_len)
    const attnElements = 2 * attnLayers * KV * headDim * seqLen;
    return {
      seqLen,
      label,
      fp16Bytes: attnElements * 2 + linearStateFp16,
      fp8Bytes: attnElements * 1 + Math.floor(linearStateFp16 / 2),
      linearStateFp16,
    };
  });
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}
