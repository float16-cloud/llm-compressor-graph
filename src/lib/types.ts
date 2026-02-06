export type LayerType = 'embedding' | 'attention' | 'mlp' | 'norm' | 'head' | 'vision' | 'group';

export interface LayerNode {
  id: string;
  name: string;
  fullPath: string;
  type: LayerType;
  children: LayerNode[];
  isSelectable: boolean;
  paramCount?: number;
}

export interface ModelConfig {
  model_type?: string;
  architectures?: string[];
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  hidden_size?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  // Hybrid attention (e.g. Qwen3-Next: full attention every N layers, rest linear)
  full_attention_interval?: number;
  // Linear/DeltaNet attention fields
  linear_num_key_heads?: number;
  linear_key_head_dim?: number;
  linear_num_value_heads?: number;
  linear_value_head_dim?: number;
  // Vision model fields
  vision_config?: {
    num_hidden_layers?: number;
    depth?: number;
    hidden_size?: number;
    num_attention_heads?: number;
    num_heads?: number;
    intermediate_size?: number;
  };
  text_config?: {
    num_hidden_layers?: number;
    hidden_size?: number;
    num_attention_heads?: number;
    num_key_value_heads?: number;
    head_dim?: number;
    intermediate_size?: number;
    vocab_size?: number;
    model_type?: string;
  };
}

export interface SafetensorsIndex {
  weight_map: Record<string, string>;
  metadata?: Record<string, unknown>;
}

export interface FetchError {
  status: number;
  message: string;
}
