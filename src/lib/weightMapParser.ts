import type { LayerNode, LayerType, SafetensorsIndex } from './types';

const WEIGHT_SUFFIXES = ['.weight', '.bias', '.scales', '.zero_point', '.weight_scale'];

function stripWeightSuffix(key: string): string {
  for (const suffix of WEIGHT_SUFFIXES) {
    if (key.endsWith(suffix)) {
      return key.slice(0, -suffix.length);
    }
  }
  return key;
}

function classifySegment(segment: string, fullPath: string): LayerType {
  const s = segment.toLowerCase();
  const fp = fullPath.toLowerCase();

  if (s.includes('embed') || s === 'wte' || s === 'wpe') return 'embedding';
  // Check norm BEFORE attention since names like "post_attention_layernorm" contain "attention"
  if (s.includes('norm') || s.includes('layernorm') || s === 'ln_1' || s === 'ln_2' ||
      s === 'ln_f' || s === 'final_layer_norm' || s === 'input_layernorm' ||
      s === 'post_attention_layernorm' || s === 'layer_norm1' || s === 'layer_norm2') return 'norm';
  if (s === 'lm_head' || s === 'score' || s === 'classifier') return 'head';
  if (s.includes('self_attn') || s.includes('attention') || s.includes('attn') ||
      s === 'q_proj' || s === 'k_proj' || s === 'v_proj' || s === 'o_proj' ||
      s === 'qkv_proj' || s === 'query' || s === 'key' || s === 'value') {
    if (fp.includes('attn') || fp.includes('attention') || fp.includes('self_attn')) {
      return 'attention';
    }
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj', 'query', 'key', 'value'].includes(s)) {
      return 'attention';
    }
  }
  if (s.includes('mlp') || s === 'gate_proj' || s === 'up_proj' || s === 'down_proj' ||
      s === 'fc1' || s === 'fc2' || s === 'c_fc' || s === 'c_proj' ||
      s === 'gate_up_proj' || s === 'dense_h_to_4h' || s === 'dense_4h_to_h') {
    if (fp.includes('mlp') || ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2',
        'c_fc', 'gate_up_proj', 'dense_h_to_4h', 'dense_4h_to_h'].includes(s)) {
      return 'mlp';
    }
  }
  if (fp.includes('vision') || fp.includes('visual') || fp.includes('image_encoder') ||
      fp.includes('vit') || fp.includes('clip')) return 'vision';
  return 'group';
}

interface TrieNode {
  children: Map<string, TrieNode>;
  isLeaf: boolean;
  fullPath: string;
}

function buildTrie(modulePaths: string[]): TrieNode {
  const root: TrieNode = { children: new Map(), isLeaf: false, fullPath: '' };

  for (const path of modulePaths) {
    const segments = path.split('.');
    let current = root;
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (!current.children.has(seg)) {
        current.children.set(seg, {
          children: new Map(),
          isLeaf: false,
          fullPath: segments.slice(0, i + 1).join('.'),
        });
      }
      current = current.children.get(seg)!;
    }
    current.isLeaf = true;
  }

  return root;
}

function trieToLayerNodes(trie: TrieNode): LayerNode[] {
  const nodes: LayerNode[] = [];

  for (const [name, child] of trie.children) {
    const type = classifySegment(name, child.fullPath);
    const childNodes = trieToLayerNodes(child);

    const node: LayerNode = {
      id: child.fullPath,
      name,
      fullPath: child.fullPath,
      type: childNodes.length > 0 ? (type === 'group' ? type : type) : type,
      children: childNodes,
      isSelectable: child.isLeaf && childNodes.length === 0,
    };

    // If this is a group with children, inherit type from classification
    if (childNodes.length > 0 && type !== 'group') {
      node.type = type;
    }

    nodes.push(node);
  }

  return nodes;
}

// Collapse numbered layers into a grouped structure
function collapseNumberedLayers(nodes: LayerNode[]): LayerNode[] {
  // Check if there are numbered children that should be grouped
  const numberedPattern = /^\d+$/;
  const numbered: LayerNode[] = [];
  const others: LayerNode[] = [];

  for (const node of nodes) {
    if (numberedPattern.test(node.name)) {
      numbered.push(node);
    } else {
      others.push(node);
    }
  }

  // Recursively collapse children
  const result = others.map(node => ({
    ...node,
    children: collapseNumberedLayers(node.children),
  }));

  // Sort numbered nodes numerically and add them
  numbered.sort((a, b) => parseInt(a.name, 10) - parseInt(b.name, 10));
  for (const node of numbered) {
    result.push({
      ...node,
      children: collapseNumberedLayers(node.children),
    });
  }

  return result;
}

export function parseWeightMap(index: SafetensorsIndex): LayerNode[] {
  const keys = Object.keys(index.weight_map);
  const modulePaths = [...new Set(keys.map(stripWeightSuffix))];
  modulePaths.sort();

  const trie = buildTrie(modulePaths);
  const tree = trieToLayerNodes(trie);
  return collapseNumberedLayers(tree);
}

function getForwardPassPriority(name: string): number {
  const s = name.toLowerCase();

  // Priority 0: Embeddings
  if (s.includes('embed') || s === 'wte' || s === 'wpe' || s === 'patch_embed' || s === 'pos_embed') return 0;

  // Priority 1: Vision modules
  if (s.includes('visual') || s.includes('vision') || s === 'vit' || s === 'clip' || s === 'image_encoder') return 1;

  // Priority 2: Merger / connector / projector
  if (s === 'merger' || s === 'connector' || s === 'projector' || s === 'multi_modal_projector' || s === 'deepstack') return 2;

  // Priority 3: Language model / generic model
  if (s === 'language_model' || s === 'model' || s === 'transformer') return 3;

  // Priority 4: Layer containers and numbered layers
  if (s === 'layers' || s === 'blocks' || s === 'h' || s === 'encoder' || /^\d+$/.test(s)) return 4;

  // Priority 5: Within-layer ordering
  if (s === 'input_layernorm' || s === 'layer_norm1') return 5.0;
  if (s === 'self_attn' || s === 'attention' || s === 'attn') return 5.1;
  if (s === 'post_attention_layernorm' || s === 'layer_norm2') return 5.2;
  if (s === 'mlp') return 5.3;

  // Priority 6: Final norms
  if (s === 'norm' || s === 'ln_f' || s === 'final_layer_norm' || s === 'final_layernorm') return 6;

  // Priority 7: Output heads
  if (s === 'lm_head' || s === 'classifier' || s === 'score') return 7;

  // Everything else
  return 99;
}

export function sortForwardPass(nodes: LayerNode[]): LayerNode[] {
  return nodes
    .slice()
    .sort((a, b) => getForwardPassPriority(a.name) - getForwardPassPriority(b.name))
    .map(node => ({
      ...node,
      children: sortForwardPass(node.children),
    }));
}

export function attachParamCounts(
  nodes: LayerNode[],
  tensorParamCounts: Map<string, number>,
): LayerNode[] {
  // Build modulePath → total params (a module can have .weight, .bias, etc.)
  const moduleParams = new Map<string, number>();
  for (const [tensorName, count] of tensorParamCounts) {
    const modulePath = stripWeightSuffix(tensorName);
    moduleParams.set(modulePath, (moduleParams.get(modulePath) ?? 0) + count);
  }

  function attach(nodes: LayerNode[]): LayerNode[] {
    return nodes.map(node => {
      if (node.children.length === 0) {
        return { ...node, paramCount: moduleParams.get(node.fullPath) ?? 0 };
      }
      const children = attach(node.children);
      const paramCount = children.reduce((sum, c) => sum + (c.paramCount ?? 0), 0);
      return { ...node, children, paramCount };
    });
  }

  return attach(nodes);
}

export function collectLeafParamCounts(nodes: LayerNode[]): Map<string, number> {
  const map = new Map<string, number>();
  function walk(node: LayerNode) {
    if (node.isSelectable && node.paramCount != null) {
      map.set(node.fullPath, node.paramCount);
    }
    for (const child of node.children) {
      walk(child);
    }
  }
  for (const node of nodes) {
    walk(node);
  }
  return map;
}

export function collectSelectablePaths(nodes: LayerNode[]): string[] {
  const paths: string[] = [];
  function walk(node: LayerNode) {
    if (node.isSelectable) {
      paths.push(node.fullPath);
    }
    for (const child of node.children) {
      walk(child);
    }
  }
  for (const node of nodes) {
    walk(node);
  }
  return paths;
}
