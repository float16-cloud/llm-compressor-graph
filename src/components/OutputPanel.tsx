import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useSelectionStore } from '../store/useSelectionStore';
import { optimizeSelection } from '../lib/regexOptimizer';
import { formatIgnoreList, formatRecipe, parseIgnoreItems, type ModifierType, type SchemeType, type KVCacheSchemeType } from '../lib/formatOutput';
import { formatBytes } from '../lib/estimateKVCache';

const MODIFIERS: ModifierType[] = ['GPTQModifier', 'QuantizationModifier', 'SmoothQuantModifier'];
const SCHEMES: SchemeType[] = ['W4A16', 'W8A16', 'FP8', 'FP8_BLOCK'];
const KV_CACHE_SCHEMES: { value: KVCacheSchemeType; label: string }[] = [
  { value: 'none', label: 'No KV Cache Quant' },
  { value: 'FP8_TENSOR', label: 'FP8 per-tensor' },
  { value: 'FP8_HEAD', label: 'FP8 per-head' },
  { value: 'INT8_TENSOR', label: 'INT8 per-tensor' },
];

function formatParamCount(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return count.toString();
}

export function OutputPanel() {
  const { selectedPaths, allSelectablePaths, paramCountMap, setSelectedPaths } = useSelectionStore();
  const [tab, setTab] = useState<'ignore' | 'recipe'>('ignore');
  const [useRegex, setUseRegex] = useState(true);
  const [modifier, setModifier] = useState<ModifierType>('GPTQModifier');
  const [scheme, setScheme] = useState<SchemeType>('W4A16');
  const [kvCacheScheme, setKvCacheScheme] = useState<KVCacheSchemeType>('none');
  const [copied, setCopied] = useState(false);
  const [editedText, setEditedText] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const selectedArray = useMemo(() => [...selectedPaths], [selectedPaths]);

  const { explicit, optimized } = useMemo(
    () => optimizeSelection(selectedArray, allSelectablePaths),
    [selectedArray, allSelectablePaths],
  );

  const items = useRegex ? optimized : explicit;

  const output = useMemo(() => {
    if (tab === 'ignore') return formatIgnoreList(items);
    return formatRecipe(items, modifier, scheme, kvCacheScheme);
  }, [tab, items, modifier, scheme, kvCacheScheme]);

  // Clear edits when tab/options change
  useEffect(() => {
    setEditedText(null);
  }, [tab, useRegex, modifier, scheme, kvCacheScheme]);

  const isEdited = editedText !== null && editedText !== output;

  // Quantization size estimates
  const quantEstimates = useMemo(() => {
    if (paramCountMap.size === 0) return null;

    let totalParams = 0;
    let ignoredParams = 0;

    for (const [path, count] of paramCountMap) {
      totalParams += count;
      if (selectedPaths.has(path)) {
        ignoredParams += count;
      }
    }

    const quantizedParams = totalParams - ignoredParams;

    // FP16: all params × 2 bytes
    const fp16Bytes = totalParams * 2;
    // W8A16: ignored at FP16 + quantized at 8-bit
    const w8Bytes = ignoredParams * 2 + quantizedParams * 1;
    // W4A16: ignored at FP16 + quantized at 4-bit
    const w4Bytes = ignoredParams * 2 + quantizedParams * 0.5;

    return {
      totalParams,
      ignoredParams,
      quantizedParams,
      fp16Bytes,
      w8Bytes,
      w4Bytes,
    };
  }, [paramCountMap, selectedPaths]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(editedText ?? output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleApply = useCallback(() => {
    if (editedText === null) return;
    const resolved = parseIgnoreItems(editedText, allSelectablePaths);
    setSelectedPaths(resolved);
    setEditedText(null);
  }, [editedText, allSelectablePaths, setSelectedPaths]);

  const handleReset = useCallback(() => {
    setEditedText(null);
  }, []);

  const displayText = editedText ?? output;
  const isEmpty = selectedPaths.size === 0 && editedText === null;

  return (
    <div className="flex flex-col h-full panel p-4">
      {/* Tabs + counter */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex gap-0.5 p-0.5 bg-surface-2 rounded-lg border border-border">
          <button
            onClick={() => setTab('ignore')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 cursor-pointer ${
              tab === 'ignore'
                ? 'bg-surface-4 text-txt shadow-sm'
                : 'text-txt-3 hover:text-txt-2'
            }`}
          >
            Ignore List
          </button>
          <button
            onClick={() => setTab('recipe')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 cursor-pointer ${
              tab === 'recipe'
                ? 'bg-surface-4 text-txt shadow-sm'
                : 'text-txt-3 hover:text-txt-2'
            }`}
          >
            Full Recipe
          </button>
        </div>
        <span className="text-[11px] text-txt-3 font-mono tabular-nums">
          <span className="text-accent">{selectedPaths.size}</span>
          <span className="text-txt-3/50"> / </span>
          {allSelectablePaths.length}
        </span>
      </div>

      {/* Options row */}
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <label className="flex items-center gap-1.5 text-xs text-txt-2 cursor-pointer">
          <input
            type="checkbox"
            checked={useRegex}
            onChange={(e) => setUseRegex(e.target.checked)}
            className="accent-[#0d9373] cursor-pointer"
          />
          Regex optimization
        </label>

        {tab === 'recipe' && (
          <>
            <select
              value={modifier}
              onChange={(e) => setModifier(e.target.value as ModifierType)}
              className="text-xs bg-surface-2 border border-border rounded-lg px-2 py-1.5 text-txt font-mono glow-focus transition-all duration-200 cursor-pointer"
            >
              {MODIFIERS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <select
              value={scheme}
              onChange={(e) => setScheme(e.target.value as SchemeType)}
              className="text-xs bg-surface-2 border border-border rounded-lg px-2 py-1.5 text-txt font-mono glow-focus transition-all duration-200 cursor-pointer"
            >
              {SCHEMES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select
              value={kvCacheScheme}
              onChange={(e) => setKvCacheScheme(e.target.value as KVCacheSchemeType)}
              className={`text-xs border rounded-lg px-2 py-1.5 font-mono glow-focus transition-all duration-200 cursor-pointer ${
                kvCacheScheme !== 'none'
                  ? 'bg-layer-embed/10 border-layer-embed/25 text-layer-embed'
                  : 'bg-surface-2 border-border text-txt'
              }`}
            >
              {KV_CACHE_SCHEMES.map(k => <option key={k.value} value={k.value}>{k.label}</option>)}
            </select>
          </>
        )}
      </div>

      {/* Quant size estimates */}
      {quantEstimates && quantEstimates.totalParams > 0 && (
        <div className="mb-3 px-3 py-2.5 bg-surface-2/60 border border-border rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] text-txt-3 uppercase tracking-widest font-mono">Estimated Size</span>
            <span className="text-[10px] text-txt-3/50 font-mono">
              {formatParamCount(quantEstimates.totalParams)} params
              {quantEstimates.ignoredParams > 0 && (
                <> &middot; {formatParamCount(quantEstimates.ignoredParams)} kept FP16</>
              )}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-surface-1 rounded-lg px-2.5 py-2 border border-border">
              <div className="text-txt-3 text-[10px] mb-1 font-mono">FP16</div>
              <div className="text-txt font-mono font-medium">{formatBytes(quantEstimates.fp16Bytes)}</div>
            </div>
            <div className="bg-surface-1 rounded-lg px-2.5 py-2 border border-border">
              <div className="text-txt-3 text-[10px] mb-1 font-mono">W8A16</div>
              <div className="text-txt font-mono font-medium">{formatBytes(quantEstimates.w8Bytes)}</div>
            </div>
            <div className="bg-surface-1 rounded-lg px-2.5 py-2 border border-border">
              <div className="text-txt-3 text-[10px] mb-1 font-mono">W4A16</div>
              <div className="text-txt font-mono font-medium">{formatBytes(quantEstimates.w4Bytes)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Code output area */}
      <div className="relative flex-1 min-h-0">
        {isEmpty ? (
          <pre className="h-full overflow-auto p-4 bg-surface-1 border border-border rounded-xl text-xs text-txt-3/50 font-mono flex items-center justify-center">
            Select layers to generate output...
          </pre>
        ) : (
          <textarea
            ref={textareaRef}
            value={displayText}
            onChange={(e) => setEditedText(e.target.value)}
            spellCheck={false}
            className={`h-full w-full resize-none overflow-auto p-4 bg-surface-1 border rounded-xl text-xs text-txt-2 font-mono whitespace-pre break-all transition-all duration-200 ${
              isEdited
                ? 'border-accent/40 focus:outline-none focus:ring-2 focus:ring-accent/15 focus:border-accent'
                : 'border-border focus:outline-none focus:ring-2 focus:ring-surface-4 focus:border-border-mid'
            }`}
          />
        )}
        {!isEmpty && (
          <div className="absolute top-2.5 right-2.5 flex gap-1.5">
            {isEdited && (
              <>
                <button
                  onClick={handleApply}
                  className="px-3 py-1.5 text-xs bg-accent text-canvas font-semibold rounded-lg hover:bg-accent-bright transition-all duration-200 cursor-pointer"
                >
                  Apply
                </button>
                <button
                  onClick={handleReset}
                  className="px-3 py-1.5 text-xs bg-surface-3 text-txt-2 border border-border rounded-lg hover:bg-surface-4 hover:text-txt transition-all duration-200 cursor-pointer"
                >
                  Reset
                </button>
              </>
            )}
            <button
              onClick={handleCopy}
              className={`px-3 py-1.5 text-xs border rounded-lg transition-all duration-200 cursor-pointer ${
                copied
                  ? 'bg-accent/15 border-accent/25 text-accent'
                  : 'bg-surface-3 text-txt-2 border-border hover:bg-surface-4 hover:text-txt'
              }`}
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
