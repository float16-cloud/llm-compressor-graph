import type { LayerNode, ModelConfig } from './types';
import { parseWeightMap, collectSelectablePaths } from './weightMapParser';

interface ArchPattern {
  test: (config: ModelConfig) => boolean;
  generate: (config: ModelConfig) => string[];
}

function llamaLikePaths(config: ModelConfig, prefix = 'model'): string[] {
  const numLayers = config.num_hidden_layers ?? config.text_config?.num_hidden_layers ?? 32;
  const paths: string[] = [
    `${prefix}.embed_tokens`,
    `${prefix}.norm`,
  ];

  for (let i = 0; i < numLayers; i++) {
    const base = `${prefix}.layers.${i}`;
    paths.push(
      `${base}.self_attn.q_proj`,
      `${base}.self_attn.k_proj`,
      `${base}.self_attn.v_proj`,
      `${base}.self_attn.o_proj`,
      `${base}.mlp.gate_proj`,
      `${base}.mlp.up_proj`,
      `${base}.mlp.down_proj`,
      `${base}.input_layernorm`,
      `${base}.post_attention_layernorm`,
    );
  }

  paths.push('lm_head');
  return paths;
}

function qwenPaths(config: ModelConfig): string[] {
  return llamaLikePaths(config, 'model');
}

function phi3Paths(config: ModelConfig): string[] {
  const numLayers = config.num_hidden_layers ?? 32;
  const paths: string[] = [
    'model.embed_tokens',
    'model.norm',
  ];

  for (let i = 0; i < numLayers; i++) {
    const base = `model.layers.${i}`;
    paths.push(
      `${base}.self_attn.qkv_proj`,
      `${base}.self_attn.o_proj`,
      `${base}.mlp.gate_up_proj`,
      `${base}.mlp.down_proj`,
      `${base}.input_layernorm`,
      `${base}.post_attention_layernorm`,
    );
  }

  paths.push('lm_head');
  return paths;
}

function qwen3NextPaths(config: ModelConfig): string[] {
  const numLayers = config.num_hidden_layers ?? 48;
  const interval = config.full_attention_interval ?? 4;
  const paths: string[] = [
    'model.embed_tokens',
    'model.norm',
  ];

  for (let i = 0; i < numLayers; i++) {
    const base = `model.layers.${i}`;
    const isFullAttn = (i + 1) % interval === 0;

    paths.push(`${base}.input_layernorm`);

    if (isFullAttn) {
      // Gated Attention layer
      paths.push(
        `${base}.self_attn.q_proj`,
        `${base}.self_attn.k_proj`,
        `${base}.self_attn.v_proj`,
        `${base}.self_attn.o_proj`,
      );
    } else {
      // Gated DeltaNet (linear attention) layer
      paths.push(
        `${base}.linear_attn.in_proj_qkvz`,
        `${base}.linear_attn.in_proj_ba`,
        `${base}.linear_attn.conv1d`,
        `${base}.linear_attn.A_log`,
        `${base}.linear_attn.dt_bias`,
        `${base}.linear_attn.norm`,
        `${base}.linear_attn.out_proj`,
      );
    }

    paths.push(
      `${base}.mlp.gate_proj`,
      `${base}.mlp.up_proj`,
      `${base}.mlp.down_proj`,
    );
  }

  paths.push('lm_head');
  return paths;
}

function gptNeoxPaths(config: ModelConfig): string[] {
  const numLayers = config.num_hidden_layers ?? 32;
  const paths: string[] = [
    'gpt_neox.embed_in',
    'gpt_neox.final_layer_norm',
  ];

  for (let i = 0; i < numLayers; i++) {
    const base = `gpt_neox.layers.${i}`;
    paths.push(
      `${base}.attention.query_key_value`,
      `${base}.attention.dense`,
      `${base}.mlp.dense_h_to_4h`,
      `${base}.mlp.dense_4h_to_h`,
      `${base}.input_layernorm`,
      `${base}.post_attention_layernorm`,
    );
  }

  paths.push('embed_out');
  return paths;
}

function addVisionPaths(config: ModelConfig): string[] {
  if (!config.vision_config) return [];

  const vc = config.vision_config;
  const numLayers = vc.num_hidden_layers ?? vc.depth ?? 24;
  const paths: string[] = [
    'visual_model.embeddings.patch_embedding',
    'visual_model.embeddings.position_embedding',
    'visual_model.layernorm',
  ];

  for (let i = 0; i < numLayers; i++) {
    const base = `visual_model.encoder.layers.${i}`;
    paths.push(
      `${base}.self_attn.q_proj`,
      `${base}.self_attn.k_proj`,
      `${base}.self_attn.v_proj`,
      `${base}.self_attn.out_proj`,
      `${base}.mlp.fc1`,
      `${base}.mlp.fc2`,
      `${base}.layer_norm1`,
      `${base}.layer_norm2`,
    );
  }

  return paths;
}

const architecturePatterns: ArchPattern[] = [
  {
    test: (c) => {
      const modelType = c.model_type ?? c.text_config?.model_type ?? '';
      return ['llama', 'mistral', 'gemma', 'gemma2', 'cohere', 'deepseek'].includes(modelType);
    },
    generate: (c) => llamaLikePaths(c),
  },
  {
    test: (c) => {
      const modelType = c.model_type ?? c.text_config?.model_type ?? '';
      return ['qwen2', 'qwen2_moe', 'qwen', 'qwen2_vl', 'qwen3', 'qwen3_vl'].includes(modelType);
    },
    generate: (c) => qwenPaths(c),
  },
  {
    test: (c) => {
      const modelType = c.model_type ?? c.text_config?.model_type ?? '';
      return modelType === 'qwen3_next';
    },
    generate: (c) => qwen3NextPaths(c),
  },
  {
    test: (c) => c.model_type === 'phi3' || c.model_type === 'phi',
    generate: (c) => phi3Paths(c),
  },
  {
    test: (c) => c.model_type === 'gpt_neox',
    generate: (c) => gptNeoxPaths(c),
  },
];

export function buildTreeFromConfig(config: ModelConfig): { tree: LayerNode[]; selectablePaths: string[] } | null {
  let paths: string[] = [];

  for (const pattern of architecturePatterns) {
    if (pattern.test(config)) {
      paths = pattern.generate(config);
      break;
    }
  }

  // Fallback: generic llama-like if we have num_hidden_layers (top-level or in text_config)
  if (paths.length === 0 && (config.num_hidden_layers || config.text_config?.num_hidden_layers)) {
    paths = llamaLikePaths(config);
  }

  // Add vision paths if applicable
  paths = [...paths, ...addVisionPaths(config)];

  if (paths.length === 0) return null;

  const fakeIndex = {
    weight_map: Object.fromEntries(paths.map(p => [`${p}.weight`, 'model.safetensors'])),
  };

  const tree = parseWeightMap(fakeIndex);
  const selectablePaths = collectSelectablePaths(tree);

  return { tree, selectablePaths };
}
