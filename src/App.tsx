import { useState, useCallback, useMemo, useRef } from 'react';
import type { LayerNode, ModelConfig } from './lib/types';
import { fetchConfig, fetchSafetensorsIndex, fetchTensorParamCounts } from './lib/api';
import { parseWeightMap, collectSelectablePaths, collectLeafParamCounts, sortForwardPass, attachParamCounts } from './lib/weightMapParser';
import { buildTreeFromConfig } from './lib/configParser';
import { estimateParamCounts } from './lib/estimateParams';
import { useSelectionStore } from './store/useSelectionStore';
import { exportTreeAsPng } from './lib/exportPng';
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

  const treePanelRef = useRef<HTMLDivElement>(null);

  const handleExportPng = useCallback(async () => {
    if (!treePanelRef.current) return;
    await exportTreeAsPng({
      element: treePanelRef.current,
      filename: modelId ? `${modelId.replace(/\//g, '-')}-architecture` : 'architecture-tree',
      backgroundColor: '#ffffff',
      modelName: modelId,
    });
  }, [modelId]);

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

          {/* GitHub link - right side */}
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-txt-3/60 font-mono tracking-widest uppercase hidden lg:block">
              Open Source
            </span>
            <a
              href="https://github.com/float16-cloud/llm-compressor-graph"
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border hover:border-border-mid bg-surface-1/50 hover:bg-surface-1 transition-all duration-200"
              title="View on GitHub"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="text-txt-3 group-hover:text-txt transition-colors duration-200"
              >
                <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
              </svg>
              <span className="text-[12px] font-medium text-txt-2 group-hover:text-txt transition-colors duration-200 hidden sm:inline">
                GitHub
              </span>
            </a>
          </div>
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
                <BatchToolbar maxLayerIndex={maxLayerIndex} sortOrder={sortOrder} onSortOrderChange={setSortOrder} onExportPng={handleExportPng} modelId={modelId} />
                <div ref={treePanelRef} className="panel p-3 max-h-[calc(100vh-360px)] overflow-y-auto">
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
