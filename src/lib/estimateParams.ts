import type { ModelConfig, LayerNode } from './types';

interface TextDims {
  H: number;
  I: number;
  V: number;
  N: number;
  KV: number;
  headDim: number;
  // Linear/DeltaNet attention dims (optional, for hybrid models)
  linKeyHeads?: number;
  linKeyDim?: number;
  linValHeads?: number;
  linValDim?: number;
}

interface VisionDims {
  VH: number;
  VI: number;
  VN: number;
}

const NORM_NAMES = new Set([
  'input_layernorm', 'post_attention_layernorm', 'norm', 'ln_f',
  'final_layernorm', 'layer_norm', 'layernorm', 'ln_1', 'ln_2',
]);

function isVisionPath(fullPath: string): boolean {
  return /visual|vision/i.test(fullPath);
}

function isNormNode(name: string): boolean {
  if (NORM_NAMES.has(name)) return true;
  if (/norm/i.test(name)) return true;
  return false;
}

function estimateLeafText(name: string, d: TextDims): number | undefined {
  switch (name) {
    case 'embed_tokens':
    case 'embed_in':
    case 'wte':
      return d.V * d.H;
    case 'q_proj':
      return d.H * (d.headDim * d.N);
    case 'k_proj':
      return d.H * (d.headDim * d.KV);
    case 'v_proj':
      return d.H * (d.headDim * d.KV);
    case 'o_proj':
    case 'dense':
      return (d.headDim * d.N) * d.H;
    case 'qkv_proj':
    case 'query_key_value':
      return d.H * (d.headDim * (d.N + 2 * d.KV));
    case 'gate_proj':
      return d.H * d.I;
    case 'up_proj':
      return d.H * d.I;
    case 'gate_up_proj':
      return d.H * 2 * d.I;
    case 'down_proj':
      return d.I * d.H;
    case 'dense_h_to_4h':
      return d.H * d.I;
    case 'dense_4h_to_h':
      return d.I * d.H;
    case 'lm_head':
    case 'embed_out':
    case 'classifier':
    case 'score':
      return d.H * d.V;
    // DeltaNet / linear attention projections
    case 'in_proj_qkvz':
      // Projects H → Q + K + V + Z (gate)
      // Q,K: linKeyHeads * linKeyDim each; V,Z: linValHeads * linValDim each
      if (d.linKeyHeads && d.linKeyDim && d.linValHeads && d.linValDim) {
        const outDim = 2 * d.linKeyHeads * d.linKeyDim + 2 * d.linValHeads * d.linValDim;
        return d.H * outDim;
      }
      return undefined;
    case 'in_proj_ba':
      // Projects H → B + A gating (same dims as key space)
      if (d.linKeyHeads && d.linKeyDim) {
        return d.H * (2 * d.linKeyHeads * d.linKeyDim);
      }
      return undefined;
    case 'out_proj':
      // Projects value output → H
      if (d.linValHeads && d.linValDim) {
        return d.linValHeads * d.linValDim * d.H;
      }
      return undefined;
    case 'conv1d':
      // 1D conv: channels * kernel_size (small)
      if (d.linKeyHeads && d.linKeyDim) {
        const convKernel = 4; // linear_conv_kernel_dim default
        return d.linKeyHeads * d.linKeyDim * convKernel;
      }
      return undefined;
    case 'A_log':
      // Decay parameter: one per key head
      return d.linKeyHeads ?? undefined;
    case 'dt_bias':
      // Delta timestep bias: one per key head
      return d.linKeyHeads ?? undefined;
    default:
      if (isNormNode(name)) return d.H;
      return undefined;
  }
}

function estimateLeafVision(name: string, d: VisionDims): number | undefined {
  switch (name) {
    case 'patch_embedding':
    case 'position_embedding':
      return undefined; // skip — depends on patch_size/channels
    case 'q_proj':
    case 'k_proj':
    case 'v_proj':
    case 'out_proj':
      return d.VH * d.VH;
    case 'fc1':
      return d.VH * d.VI;
    case 'fc2':
      return d.VI * d.VH;
    default:
      if (isNormNode(name)) return d.VH;
      return undefined;
  }
}

function walkTree(
  nodes: LayerNode[],
  textDims: TextDims | null,
  visionDims: VisionDims | null,
): LayerNode[] {
  return nodes.map(node => {
    if (node.children.length === 0) {
      // Leaf node — estimate from formula
      const inVision = isVisionPath(node.fullPath);
      let paramCount: number | undefined;

      if (inVision && visionDims) {
        paramCount = estimateLeafVision(node.name, visionDims);
      } else if (!inVision && textDims) {
        paramCount = estimateLeafText(node.name, textDims);
      }

      if (paramCount !== undefined) {
        return { ...node, paramCount };
      }
      return node;
    }

    // Group node — recurse then sum children
    const newChildren = walkTree(node.children, textDims, visionDims);
    const sum = newChildren.reduce((acc, c) => acc + (c.paramCount ?? 0), 0);
    return {
      ...node,
      children: newChildren,
      paramCount: sum > 0 ? sum : node.paramCount,
    };
  });
}

export function estimateParamCounts(config: ModelConfig, nodes: LayerNode[]): LayerNode[] {
  // Resolve text dimensions
  const H = config.hidden_size ?? config.text_config?.hidden_size;
  const I = config.intermediate_size ?? config.text_config?.intermediate_size;
  const V = config.vocab_size ?? config.text_config?.vocab_size;
  const N = config.num_attention_heads ?? config.text_config?.num_attention_heads;

  let textDims: TextDims | null = null;
  if (H && I && V && N) {
    const KV = config.num_key_value_heads ?? config.text_config?.num_key_value_heads ?? N;
    const headDim = config.head_dim ?? config.text_config?.head_dim ?? Math.floor(H / N);
    textDims = {
      H, I, V, N, KV, headDim,
      linKeyHeads: config.linear_num_key_heads,
      linKeyDim: config.linear_key_head_dim,
      linValHeads: config.linear_num_value_heads,
      linValDim: config.linear_value_head_dim,
    };
  }

  // Resolve vision dimensions
  let visionDims: VisionDims | null = null;
  const vc = config.vision_config;
  if (vc) {
    const VH = vc.hidden_size;
    const VI = vc.intermediate_size;
    const VN = vc.num_attention_heads ?? vc.num_heads;
    if (VH && VI && VN) {
      visionDims = { VH, VI, VN };
    }
  }

  return walkTree(nodes, textDims, visionDims);
}
