import { useState, useCallback } from 'react';
import { useSelectionStore } from '../store/useSelectionStore';

function computeAutoIgnore(paths: string[], maxLayerIndex: number, stage: 1 | 2 | 3): string[] {
  const result: string[] = [];

  for (const path of paths) {
    // Stage 1: lm_head, norms, MoE gates/routers
    if (/lm_head|embed_out|classifier|score/.test(path)) { result.push(path); continue; }
    if (/norm|layernorm|ln_/.test(path)) { result.push(path); continue; }
    // MoE gate/router (NOT gate_proj which is SwiGLU MLP)
    if (/\.gate\b/.test(path) && !/gate_proj|gate_up_proj/.test(path)) { result.push(path); continue; }
    if (/shared_expert_gate/.test(path)) { result.push(path); continue; }

    if (stage < 2) continue;

    // Stage 2: first/last 3 layers + o_proj everywhere
    const layerMatch = path.match(/\.layers\.(\d+)\./);
    if (layerMatch) {
      const idx = parseInt(layerMatch[1], 10);
      if (idx <= 2 || idx >= maxLayerIndex - 2) {
        result.push(path);
        continue;
      }
    }
    if (/\.o_proj\b/.test(path)) { result.push(path); continue; }

    if (stage < 3) continue;

    // Stage 3: MLP layers in middle 30-70% depth
    if (layerMatch) {
      const idx = parseInt(layerMatch[1], 10);
      const pct = maxLayerIndex > 0 ? idx / maxLayerIndex : 0;
      if (pct >= 0.3 && pct <= 0.7) {
        if (/mlp|gate_proj|up_proj|down_proj|gate_up_proj|fc1|fc2|dense_h_to_4h|dense_4h_to_h|experts/.test(path)) {
          result.push(path);
          continue;
        }
      }
    }
  }

  return result;
}

interface BatchToolbarProps {
  maxLayerIndex: number;
  sortOrder: 'weight-file' | 'forward-pass';
  onSortOrderChange: (order: 'weight-file' | 'forward-pass') => void;
}

export function BatchToolbar({ maxLayerIndex, sortOrder, onSortOrderChange }: BatchToolbarProps) {
  const { selectAll, deselectAll, invertSelection, selectByPattern, selectRange, setSelectedPaths, allSelectablePaths } = useSelectionStore();
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(maxLayerIndex);

  const handleAutoIgnore = useCallback((stage: 1 | 2 | 3) => {
    const paths = computeAutoIgnore(allSelectablePaths, maxLayerIndex, stage);
    setSelectedPaths(paths);
  }, [allSelectablePaths, maxLayerIndex, setSelectedPaths]);

  // Update range end when maxLayerIndex changes
  if (rangeEnd > maxLayerIndex) {
    setRangeEnd(maxLayerIndex);
  }

  const quickSelects: { label: string; pattern: RegExp; color: string }[] = [
    { label: 'All Attention', pattern: /\.self_attn\.|\.attention\.|\.attn\./, color: 'bg-layer-attn/12 text-layer-attn border-layer-attn/20 hover:bg-layer-attn/20' },
    { label: 'All MLP', pattern: /mlp|gate_proj|up_proj|down_proj|fc1|fc2|dense_h_to_4h|dense_4h_to_h/, color: 'bg-layer-mlp/12 text-layer-mlp border-layer-mlp/20 hover:bg-layer-mlp/20' },
    { label: 'All Norms', pattern: /norm|ln_/, color: 'bg-layer-norm/12 text-layer-norm border-layer-norm/20 hover:bg-layer-norm/20' },
    { label: 'All Vision', pattern: /vision|visual|vit|clip|image/, color: 'bg-layer-vision/12 text-layer-vision border-layer-vision/20 hover:bg-layer-vision/20' },
  ];

  return (
    <div className="card px-4 py-3.5 space-y-3">
      {/* Sort order */}
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-txt-3 uppercase tracking-widest font-mono">Order</span>
        <div className="inline-flex rounded-lg border border-border overflow-hidden">
          <button
            onClick={() => onSortOrderChange('weight-file')}
            className={`px-3 py-1.5 text-xs font-medium transition-all duration-200 cursor-pointer ${
              sortOrder === 'weight-file'
                ? 'bg-surface-3 text-txt'
                : 'bg-surface-1 text-txt-3 hover:bg-surface-2 hover:text-txt-2'
            }`}
          >
            Weight file
          </button>
          <button
            onClick={() => onSortOrderChange('forward-pass')}
            className={`px-3 py-1.5 text-xs font-medium transition-all duration-200 border-l border-border cursor-pointer ${
              sortOrder === 'forward-pass'
                ? 'bg-surface-3 text-txt'
                : 'bg-surface-1 text-txt-3 hover:bg-surface-2 hover:text-txt-2'
            }`}
          >
            Forward pass
          </button>
        </div>
      </div>

      {/* Quick selects */}
      <div className="flex flex-wrap gap-1.5">
        {quickSelects.map(({ label, pattern, color }) => {
          const hasMatchingPaths = allSelectablePaths.some(p => pattern.test(p));
          if (!hasMatchingPaths) return null;
          return (
            <button
              key={label}
              onClick={() => selectByPattern(pattern)}
              className={`px-2.5 py-1 text-[11px] font-mono border rounded-lg transition-all duration-200 cursor-pointer ${color}`}
            >
              + {label}
            </button>
          );
        })}
      </div>

      {/* Bulk actions */}
      <div className="flex flex-wrap gap-1.5">
        <button onClick={selectAll} className="px-3 py-1.5 text-xs bg-surface-2 text-txt-2 border border-border rounded-lg hover:bg-surface-3 hover:text-txt transition-all duration-200 cursor-pointer">
          Select All
        </button>
        <button onClick={deselectAll} className="px-3 py-1.5 text-xs bg-surface-2 text-txt-2 border border-border rounded-lg hover:bg-surface-3 hover:text-txt transition-all duration-200 cursor-pointer">
          Deselect All
        </button>
        <button onClick={invertSelection} className="px-3 py-1.5 text-xs bg-surface-2 text-txt-2 border border-border rounded-lg hover:bg-surface-3 hover:text-txt transition-all duration-200 cursor-pointer">
          Invert
        </button>
      </div>

      {/* Auto-ignore presets */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[10px] text-txt-3 uppercase tracking-widest font-mono mr-1">Auto Ignore</span>
        <button
          onClick={() => handleAutoIgnore(1)}
          className="px-2.5 py-1 text-[11px] font-mono border rounded-lg transition-all duration-200 bg-layer-head/10 text-layer-head border-layer-head/20 hover:bg-layer-head/20 cursor-pointer"
          title="Protect lm_head, norms, MoE gates — most aggressive quantization"
        >
          Aggressive
        </button>
        <button
          onClick={() => handleAutoIgnore(2)}
          className="px-2.5 py-1 text-[11px] font-mono border rounded-lg transition-all duration-200 bg-layer-norm/10 text-layer-norm border-layer-norm/20 hover:bg-layer-norm/20 cursor-pointer"
          title="+ first/last 3 layers, o_proj — balanced quality/size"
        >
          Balanced
        </button>
        <button
          onClick={() => handleAutoIgnore(3)}
          className="px-2.5 py-1 text-[11px] font-mono border rounded-lg transition-all duration-200 bg-layer-mlp/10 text-layer-mlp border-layer-mlp/20 hover:bg-layer-mlp/20 cursor-pointer"
          title="+ middle MLP layers — best quality, least compression"
        >
          Conservative
        </button>
      </div>

      {/* Layer range */}
      {maxLayerIndex > 0 && (
        <div className="flex items-center gap-2 text-xs pt-1 border-t border-border">
          <span className="text-[10px] text-txt-3 uppercase tracking-widest font-mono">Range</span>
          <input
            type="number"
            min={0}
            max={maxLayerIndex}
            value={rangeStart}
            onChange={(e) => setRangeStart(Math.max(0, Math.min(parseInt(e.target.value) || 0, maxLayerIndex)))}
            className="w-14 px-2 py-1.5 bg-surface-2 border border-border rounded-lg text-txt font-mono text-center glow-focus transition-all duration-200"
          />
          <span className="text-txt-3">&rarr;</span>
          <input
            type="number"
            min={0}
            max={maxLayerIndex}
            value={rangeEnd}
            onChange={(e) => setRangeEnd(Math.max(0, Math.min(parseInt(e.target.value) || 0, maxLayerIndex)))}
            className="w-14 px-2 py-1.5 bg-surface-2 border border-border rounded-lg text-txt font-mono text-center glow-focus transition-all duration-200"
          />
          <button
            onClick={() => selectRange(rangeStart, rangeEnd)}
            className="px-3 py-1.5 bg-surface-3 text-txt-2 border border-border rounded-lg hover:bg-surface-4 hover:text-txt transition-all duration-200 cursor-pointer"
          >
            Select range
          </button>
        </div>
      )}
    </div>
  );
}
