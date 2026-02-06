import { useState, useCallback, useMemo } from 'react';
import type { LayerNode, ModelConfig } from './lib/types';
import { fetchConfig, fetchSafetensorsIndex, fetchTensorParamCounts } from './lib/api';
import { parseWeightMap, collectSelectablePaths, collectLeafParamCounts, sortForwardPass, attachParamCounts } from './lib/weightMapParser';
import { buildTreeFromConfig } from './lib/configParser';
import { estimateParamCounts } from './lib/estimateParams';
import { useSelectionStore } from './store/useSelectionStore';
import { ModelInput } from './components/ModelInput';
import { ModelSummary } from './components/ModelSummary';
import { BatchToolbar } from './components/BatchToolbar';
import { ArchitectureTree } from './components/ArchitectureTree';
import { OutputPanel } from './components/OutputPanel';

function getMaxLayerIndex(paths: string[]): number {
  let max = 0;
  const layerPattern = /\.(\d+)\./;
  for (const p of paths) {
    const match = p.match(layerPattern);
    if (match) {
      const idx = parseInt(match[1], 10);
      if (idx > max) max = idx;
    }
  }
  return max;
}

export default function App() {
  const [tree, setTree] = useState<LayerNode[]>([]);
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [modelId, setModelId] = useState<string>('');
  const [source, setSource] = useState<'safetensors' | 'config'>('safetensors');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<'weight-file' | 'forward-pass'>('forward-pass');

  const { setAllSelectablePaths, setParamCountMap, allSelectablePaths } = useSelectionStore();

  const maxLayerIndex = useMemo(() => getMaxLayerIndex(allSelectablePaths), [allSelectablePaths]);
  const displayTree = useMemo(() => sortOrder === 'forward-pass' ? sortForwardPass(tree) : tree, [tree, sortOrder]);

  const [baseTree, setBaseTree] = useState<LayerNode[]>([]);
  const [estimated, setEstimated] = useState(false);

  const handleToggleEstimate = useCallback(() => {
    if (!config) return;
    if (estimated) {
      setTree(baseTree);
      setParamCountMap(collectLeafParamCounts(baseTree));
      setEstimated(false);
    } else {
      const est = estimateParamCounts(config, baseTree);
      setTree(est);
      setParamCountMap(collectLeafParamCounts(est));
      setEstimated(true);
    }
  }, [config, estimated, baseTree, setParamCountMap]);

  const handleLoad = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    setTree([]);
    setBaseTree([]);
    setEstimated(false);
    setConfig(null);
    setModelId(id);
    setAllSelectablePaths([]);

    try {
      // Fetch config and safetensors index in parallel
      const [configResult, indexResult] = await Promise.allSettled([
        fetchConfig(id),
        fetchSafetensorsIndex(id),
      ]);

      const cfg = configResult.status === 'fulfilled' ? configResult.value : null;
      const index = indexResult.status === 'fulfilled' ? indexResult.value : null;

      if (!cfg && !index) {
        const msg = configResult.status === 'rejected'
          ? configResult.reason?.message ?? 'Failed to fetch model'
          : 'No model files found';
        setError(msg);
        setLoading(false);
        return;
      }

      if (cfg) setConfig(cfg);

      if (index) {
        // Primary path: parse weight map
        const parsedTree = parseWeightMap(index);
        const selectablePaths = collectSelectablePaths(parsedTree);
        setTree(parsedTree);
        setAllSelectablePaths(selectablePaths);
        setSource('safetensors');

        // Fetch param counts in the background (non-blocking)
        fetchTensorParamCounts(id, index.weight_map)
          .then(tensorParams => {
            setTree(prev => {
              const updated = attachParamCounts(prev, tensorParams);
              setParamCountMap(collectLeafParamCounts(updated));
              return updated;
            });
          })
          .catch(() => {
            // Param counts are optional — silently degrade
          });
      } else if (cfg) {
        // Fallback: generate from config
        const result = buildTreeFromConfig(cfg);
        if (result) {
          setBaseTree(result.tree);
          const estimated = estimateParamCounts(cfg, result.tree);
          setTree(estimated);
          setEstimated(true);
          setAllSelectablePaths(result.selectablePaths);
          setParamCountMap(collectLeafParamCounts(estimated));
          setSource('config');
        } else {
          setError('Could not determine model architecture from config. Unsupported model type.');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  }, [setAllSelectablePaths, setParamCountMap]);

  return (
    <div className="min-h-screen text-txt font-display">
      {/* Accent top line */}
      <div className="accent-line" />

      {/* Header */}
      <header className="border-b border-border px-6 py-3">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Stacked-layers icon */}
            <div className="w-8 h-8 rounded-lg bg-accent/8 border border-accent/15 flex items-center justify-center text-accent">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 1L15 5L8 9L1 5L8 1Z" fill="currentColor" opacity="0.25" />
                <path d="M8 4.5L15 8.5L8 12.5L1 8.5L8 4.5Z" fill="currentColor" opacity="0.5" />
                <path d="M8 8L15 12L8 16L1 12L8 8Z" fill="currentColor" opacity="0.85" />
              </svg>
            </div>
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight text-txt">
                <span className="text-txt-2">LLM Compressor</span>{' '}
                Graph
              </h1>
              <p className="text-[11px] text-txt-3 mt-0.5 font-mono">
                Visualize layers &middot; Generate ignore lists for llm-compressor
              </p>
            </div>
          </div>
          <span className="text-[11px] text-txt-3/60 font-mono tracking-widest uppercase hidden sm:block">
            llm-compressor-graph
          </span>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-screen-2xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_420px] gap-6">
          {/* Left panel */}
          <div className="space-y-4 min-w-0">
            <ModelInput onLoad={handleLoad} loading={loading} error={error} />

            {config && (
              <div className="animate-fade-up" style={{ animationDelay: '0.05s' }}>
                <ModelSummary config={config} modelId={modelId} source={source} onToggleEstimate={handleToggleEstimate} estimated={estimated} />
              </div>
            )}

            {tree.length > 0 && (
              <div className="animate-fade-up space-y-4" style={{ animationDelay: '0.1s' }}>
                <BatchToolbar maxLayerIndex={maxLayerIndex} sortOrder={sortOrder} onSortOrderChange={setSortOrder} />
                <div className="panel p-3 max-h-[calc(100vh-360px)] overflow-y-auto">
                  <ArchitectureTree tree={displayTree} />
                </div>
              </div>
            )}
          </div>

          {/* Right panel (sticky) */}
          {tree.length > 0 && (
            <div className="lg:sticky lg:top-6 lg:self-start lg:h-[calc(100vh-72px)] animate-fade-up" style={{ animationDelay: '0.15s' }}>
              <OutputPanel />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
