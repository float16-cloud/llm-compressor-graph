import { useState } from 'react';

const EXAMPLE_MODELS = [
  'typhoon-ai/typhoon-ocr1.5-2b',
  'meta-llama/Llama-3.1-8B',
  'Qwen/Qwen2.5-7B',
  'mistralai/Mistral-7B-v0.3',
  'microsoft/Phi-3-mini-4k-instruct',
];

interface ModelInputProps {
  onLoad: (modelId: string) => void;
  loading: boolean;
  error: string | null;
}

export function ModelInput({ onLoad, loading, error }: ModelInputProps) {
  const [modelId, setModelId] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = modelId.trim();
    if (trimmed) onLoad(trimmed);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-txt-3">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path d="M7 12.5A5.5 5.5 0 107 1.5a5.5 5.5 0 000 11zM14.5 14.5l-3.857-3.857" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <input
            type="text"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="e.g. meta-llama/Llama-3.1-8B"
            className="w-full pl-9 pr-3 py-2.5 bg-surface-1 border border-border rounded-xl text-sm font-mono text-txt placeholder-txt-3/50 glow-focus transition-all duration-200"
            disabled={loading}
          />
        </div>
        <button
          type="submit"
          disabled={loading || !modelId.trim()}
          className="px-5 py-2.5 bg-accent text-canvas text-sm font-semibold rounded-xl hover:bg-accent-bright disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 cursor-pointer"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading
            </span>
          ) : 'Load'}
        </button>
      </form>

      <div className="flex flex-wrap gap-1.5">
        {EXAMPLE_MODELS.map((id) => (
          <button
            key={id}
            onClick={() => { setModelId(id); onLoad(id); }}
            disabled={loading}
            className="px-3 py-1.5 text-xs font-mono bg-surface-2/60 text-txt-2 border border-border rounded-lg hover:bg-surface-3 hover:text-txt hover:border-border-mid disabled:opacity-40 transition-all duration-200 cursor-pointer"
          >
            {id}
          </button>
        ))}
      </div>

      {error && (
        <div className="px-3 py-2.5 bg-layer-head/8 border border-layer-head/20 rounded-xl text-sm text-layer-head animate-fade-up">
          {error}
        </div>
      )}
    </div>
  );
}
