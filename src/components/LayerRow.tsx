import type { LayerNode, LayerType } from '../lib/types';

function formatParamCount(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return count.toString();
}

function getAttnBadge(node: LayerNode): { label: string; className: string } | null {
  if (node.type !== 'group' || node.children.length === 0) return null;
  const childNames = node.children.map(c => c.name);
  if (childNames.includes('linear_attn')) {
    return { label: 'DeltaNet', className: 'bg-cyan-400/10 text-cyan-400 border-cyan-400/25' };
  }
  if (childNames.includes('self_attn')) {
    return { label: 'Full Attn', className: 'bg-indigo-400/10 text-indigo-400 border-indigo-400/25' };
  }
  return null;
}

const TYPE_COLORS: Record<LayerType, string> = {
  attention: 'bg-layer-attn/10 text-layer-attn border-layer-attn/25',
  mlp: 'bg-layer-mlp/10 text-layer-mlp border-layer-mlp/25',
  norm: 'bg-layer-norm/10 text-layer-norm border-layer-norm/25',
  embedding: 'bg-layer-embed/10 text-layer-embed border-layer-embed/25',
  head: 'bg-layer-head/10 text-layer-head border-layer-head/25',
  vision: 'bg-layer-vision/10 text-layer-vision border-layer-vision/25',
  group: 'bg-surface-3/50 text-txt-3 border-border-mid',
};

const TYPE_CHECKBOX: Record<LayerType, string> = {
  attention: 'accent-[#0284c7]',
  mlp: 'accent-[#059669]',
  norm: 'accent-[#d97706]',
  embedding: 'accent-[#7c3aed]',
  head: 'accent-[#dc2626]',
  vision: 'accent-[#ea580c]',
  group: 'accent-[#6b7280]',
};

interface LayerRowProps {
  node: LayerNode;
  depth: number;
  expanded: boolean;
  onToggleExpand: () => void;
  checked: boolean | 'indeterminate';
  onToggleCheck: () => void;
  hasChildren: boolean;
}

export function LayerRow({
  node,
  depth,
  expanded,
  onToggleExpand,
  checked,
  onToggleCheck,
  hasChildren,
}: LayerRowProps) {
  const typeColor = TYPE_COLORS[node.type];
  const checkboxColor = TYPE_CHECKBOX[node.type];
  const attnBadge = getAttnBadge(node);

  return (
    <div
      className="flex items-center gap-1.5 py-[3px] hover:bg-surface-2/50 rounded-md group transition-colors duration-150"
      style={{ paddingLeft: `${depth * 18 + 6}px` }}
      title={node.fullPath}
    >
      {hasChildren ? (
        <button
          onClick={onToggleExpand}
          className="w-4 h-4 flex items-center justify-center text-txt-3 hover:text-accent transition-colors duration-150 flex-shrink-0 cursor-pointer"
        >
          <svg
            className={`w-3 h-3 transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`}
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
          </svg>
        </button>
      ) : (
        <span className="w-4 flex-shrink-0" />
      )}

      <input
        type="checkbox"
        checked={checked === true}
        ref={(el) => {
          if (el) el.indeterminate = checked === 'indeterminate';
        }}
        onChange={onToggleCheck}
        className={`w-3.5 h-3.5 rounded flex-shrink-0 cursor-pointer ${checkboxColor}`}
      />

      <span className="text-xs font-mono text-txt truncate">
        {node.name}
      </span>

      <span className={`badge flex-shrink-0 ${typeColor}`}>
        {node.type}
      </span>

      {attnBadge && (
        <span className={`badge flex-shrink-0 ${attnBadge.className}`}>
          {attnBadge.label}
        </span>
      )}

      {node.paramCount != null && node.paramCount > 0 && (
        <span className="text-[10px] text-txt-3 flex-shrink-0 tabular-nums font-mono">
          {formatParamCount(node.paramCount)}
        </span>
      )}

      <span className="text-[10px] text-txt-3/40 truncate hidden group-hover:inline ml-auto font-mono">
        {node.fullPath}
      </span>
    </div>
  );
}
