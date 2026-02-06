import { useMemo } from 'react';
import type { ModelConfig } from '../lib/types';
import { estimateKVCache, formatBytes } from '../lib/estimateKVCache';

interface ModelSummaryProps {
  config: ModelConfig;
  modelId: string;
  source: 'safetensors' | 'config';
  onToggleEstimate?: () => void;
  estimated?: boolean;
}

export function ModelSummary({ config, modelId, source, onToggleEstimate, estimated }: ModelSummaryProps) {
  const modelType = config.model_type ?? config.text_config?.model_type ?? 'unknown';
  const numLayers = config.num_hidden_layers ?? config.text_config?.num_hidden_layers ?? '?';
  const hiddenSize = config.hidden_size ?? config.text_config?.hidden_size ?? '?';
  const vocabSize = config.vocab_size ?? config.text_config?.vocab_size ?? '?';
  const numHeads = config.num_attention_heads ?? config.text_config?.num_attention_heads ?? '?';
  const intermediateSize = config.intermediate_size ?? config.text_config?.intermediate_size ?? '?';
  const hasVision = !!config.vision_config;
  const visionLayers = config.vision_config?.num_hidden_layers ?? config.vision_config?.depth;
  const kvEstimates = useMemo(() => estimateKVCache(config), [config]);
  const isHybrid = (config.full_attention_interval ?? 0) > 1;
  const totalLayers = config.num_hidden_layers ?? config.text_config?.num_hidden_layers ?? 0;
  const attnLayerCount = isHybrid ? Math.floor(totalLayers / config.full_attention_interval!) : totalLayers;
  const linearLayerCount = totalLayers - attnLayerCount;

  return (
    <div className="card px-4 py-3.5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-txt truncate font-mono" title={modelId}>{modelId}</h3>
        <span className="badge bg-surface-3 text-txt-2 border-border">
          {source === 'safetensors' ? 'weight map' : 'config only'}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-8 gap-y-1.5 text-xs">
        <div className="flex justify-between">
          <span className="text-txt-3">Type</span>
          {source === 'config' && onToggleEstimate ? (
            <button
              className="text-txt font-mono underline decoration-dotted decoration-txt-3/40 underline-offset-2 hover:text-accent transition-colors cursor-pointer"
              title={estimated ? 'Click to hide estimated parameters' : 'Click to estimate parameter counts from config'}
              onClick={onToggleEstimate}
            >
              {modelType}{estimated && <span className="text-txt-3 ml-1 text-[10px]">(estimated)</span>}
            </button>
          ) : (
            <span className="text-txt font-mono">{modelType}</span>
          )}
        </div>
        <div className="flex justify-between">
          <span className="text-txt-3">Layers</span>
          <span className="text-txt font-mono">
            {numLayers}
            {isHybrid && (
              <span className="text-txt-3 text-[10px] ml-1">({attnLayerCount} attn + {linearLayerCount} linear)</span>
            )}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-txt-3">Hidden</span>
          <span className="text-txt font-mono">{hiddenSize}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-txt-3">Heads</span>
          <span className="text-txt font-mono">{numHeads}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-txt-3">Vocab</span>
          <span className="text-txt font-mono">{vocabSize}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-txt-3">FFN size</span>
          <span className="text-txt font-mono">{intermediateSize}</span>
        </div>
        {hasVision && (
          <div className="flex justify-between col-span-2">
            <span className="text-txt-3">Vision layers</span>
            <span className="text-txt font-mono">{visionLayers ?? '?'}</span>
          </div>
        )}
      </div>
      {kvEstimates && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-txt-3 uppercase tracking-widest font-mono">KV Cache</span>
            {isHybrid && (
              <span className="text-[10px] text-txt-3/50">
                {attnLayerCount} attn layers + {linearLayerCount} linear state
              </span>
            )}
          </div>
          <div className="mt-1.5 grid grid-cols-3 gap-2 text-xs">
            {kvEstimates.map((est) => (
              <div key={est.label} className="bg-surface-2/60 rounded-lg px-2.5 py-2 border border-border">
                <div className="text-txt-3 text-[10px] mb-1 font-mono">{est.label} tokens</div>
                <div className="text-txt font-mono font-medium">{formatBytes(est.fp16Bytes)}</div>
                <div className="text-txt-3 font-mono text-[10px] mt-0.5">FP8: {formatBytes(est.fp8Bytes)}</div>
                {est.linearStateFp16 > 0 && (
                  <div className="text-txt-3/50 font-mono text-[10px]">incl. {formatBytes(est.linearStateFp16)} state</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
