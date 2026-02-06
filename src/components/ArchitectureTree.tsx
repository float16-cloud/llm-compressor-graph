import { useState, useCallback, useMemo } from 'react';
import type { LayerNode } from '../lib/types';
import { useSelectionStore } from '../store/useSelectionStore';
import { LayerRow } from './LayerRow';

function collectLeafPaths(node: LayerNode): string[] {
  if (node.isSelectable) return [node.fullPath];
  const paths: string[] = [];
  for (const child of node.children) {
    paths.push(...collectLeafPaths(child));
  }
  return paths;
}

function getCheckState(
  node: LayerNode,
  selectedPaths: Set<string>,
): boolean | 'indeterminate' {
  if (node.isSelectable) {
    return selectedPaths.has(node.fullPath);
  }

  const leafPaths = collectLeafPaths(node);
  if (leafPaths.length === 0) return false;

  const selectedCount = leafPaths.filter(p => selectedPaths.has(p)).length;
  if (selectedCount === 0) return false;
  if (selectedCount === leafPaths.length) return true;
  return 'indeterminate';
}

// Determine which groups should be collapsed by default
function shouldStartCollapsed(node: LayerNode): boolean {
  // Collapse numbered layer children (e.g., layers.0, layers.1, ...)
  return /^\d+$/.test(node.name) && node.children.length > 0;
}

interface TreeNodeProps {
  node: LayerNode;
  depth: number;
}

function TreeNode({ node, depth }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(!shouldStartCollapsed(node));
  const { selectedPaths, selectPaths, deselectPaths, toggle } = useSelectionStore();

  const checkState = useMemo(
    () => getCheckState(node, selectedPaths),
    [node, selectedPaths],
  );

  const handleToggleCheck = useCallback(() => {
    if (node.isSelectable) {
      toggle(node.fullPath);
    } else {
      const leafPaths = collectLeafPaths(node);
      // If all are selected, deselect all; otherwise select all
      const allSelected = leafPaths.every(p => selectedPaths.has(p));
      if (allSelected) {
        deselectPaths(leafPaths);
      } else {
        selectPaths(leafPaths);
      }
    }
  }, [node, selectedPaths, toggle, selectPaths, deselectPaths]);

  const hasChildren = node.children.length > 0;

  return (
    <>
      <LayerRow
        node={node}
        depth={depth}
        expanded={expanded}
        onToggleExpand={() => setExpanded(!expanded)}
        checked={checkState}
        onToggleCheck={handleToggleCheck}
        hasChildren={hasChildren}
      />
      {expanded && hasChildren && (
        node.children.map(child => (
          <TreeNode key={child.id} node={child} depth={depth + 1} />
        ))
      )}
    </>
  );
}

interface ArchitectureTreeProps {
  tree: LayerNode[];
}

export function ArchitectureTree({ tree }: ArchitectureTreeProps) {
  if (tree.length === 0) {
    return (
      <div className="text-sm text-txt-3 py-6 text-center font-mono">
        No layers found.
      </div>
    );
  }

  return (
    <div className="font-mono text-sm">
      {tree.map(node => (
        <TreeNode key={node.id} node={node} depth={0} />
      ))}
    </div>
  );
}
