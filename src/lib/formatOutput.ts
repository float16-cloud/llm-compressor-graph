export type ModifierType = 'GPTQModifier' | 'QuantizationModifier' | 'SmoothQuantModifier';
export type SchemeType = 'W4A16' | 'W8A16' | 'FP8' | 'FP8_BLOCK';
export type KVCacheSchemeType = 'none' | 'FP8_TENSOR' | 'FP8_HEAD' | 'INT8_TENSOR';

/**
 * Parse ignore list text back into resolved paths.
 * Handles both literal paths and `re:` regex patterns.
 */
export function parseIgnoreItems(text: string, allSelectablePaths: string[]): string[] {
  // Extract content inside [...] (handles both ignore=[...] and recipe ignore=[...])
  const bracketMatch = text.match(/\[([^\]]*)\]/s);
  if (!bracketMatch) return [];

  const inner = bracketMatch[1];
  // Extract quoted strings
  const items: string[] = [];
  const re = /["']([^"']+)["']/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(inner)) !== null) {
    items.push(m[1]);
  }

  // Resolve each item against allSelectablePaths
  const result = new Set<string>();
  for (const item of items) {
    if (item.startsWith('re:')) {
      try {
        const pattern = new RegExp(item.slice(3));
        for (const p of allSelectablePaths) {
          if (pattern.test(p)) result.add(p);
        }
      } catch {
        // Invalid regex — skip
      }
    } else {
      // Literal path — only add if it exists
      if (allSelectablePaths.includes(item)) {
        result.add(item);
      }
    }
  }

  return [...result];
}

function formatStringList(items: string[]): string {
  if (items.length === 0) return '[]';
  const inner = items.map(item => `    "${item}",`).join('\n');
  return `[\n${inner}\n]`;
}

export function formatIgnoreList(items: string[]): string {
  return `ignore=${formatStringList(items)}`;
}

function formatKVCacheScheme(kvScheme: KVCacheSchemeType): string | null {
  switch (kvScheme) {
    case 'FP8_TENSOR':
      return `    kv_cache_scheme={
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    }`;
    case 'FP8_HEAD':
      return `    kv_cache_scheme={
        "num_bits": 8,
        "type": "float",
        "strategy": "attn_head",
        "dynamic": False,
        "symmetric": True,
    }`;
    case 'INT8_TENSOR':
      return `    kv_cache_scheme={
        "num_bits": 8,
        "type": "int",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    }`;
    default:
      return null;
  }
}

export function formatRecipe(
  items: string[],
  modifier: ModifierType,
  scheme: SchemeType,
  kvCacheScheme: KVCacheSchemeType = 'none',
): string {
  const importPath = modifier === 'SmoothQuantModifier'
    ? 'llmcompressor.modifiers.smoothquant'
    : 'llmcompressor.modifiers.quantization';

  const ignoreStr = formatStringList(items);
  const kvBlock = formatKVCacheScheme(kvCacheScheme);

  const args = [
    `    targets="Linear"`,
    `    scheme="${scheme}"`,
    `    ignore=${ignoreStr}`,
  ];

  if (kvBlock) {
    args.push(kvBlock);
  }

  const vllmNote = kvCacheScheme !== 'none'
    ? `\n\n# vLLM: load with kv_cache_dtype="${kvCacheScheme.startsWith('FP8') ? 'fp8' : 'int8'}"`
    : '';

  return `from ${importPath} import ${modifier}

recipe = ${modifier}(
${args.join(',\n')},
)${vllmNote}`;
}
