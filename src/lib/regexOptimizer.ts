function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Replace layer indices with a placeholder to find templates
function toTemplate(path: string): { template: string; index: number } | null {
  const match = path.match(/^(.+\.)(\d+)(\..+)$/);
  if (!match) return null;
  return {
    template: `${match[1]}{{INDEX}}${match[3]}`,
    index: parseInt(match[2], 10),
  };
}

interface TemplateGroup {
  template: string;
  prefix: string;
  suffix: string;
  indices: number[];
}

export function optimizeSelection(
  selectedPaths: string[],
  allSelectablePaths: string[],
): { explicit: string[]; optimized: string[] } {
  if (selectedPaths.length === 0) {
    return { explicit: [], optimized: [] };
  }

  // Group paths by template
  const templateGroups = new Map<string, TemplateGroup>();
  const nonTemplated: string[] = [];

  for (const path of selectedPaths) {
    const parsed = toTemplate(path);
    if (!parsed) {
      nonTemplated.push(path);
      continue;
    }

    const { template, index } = parsed;
    if (!templateGroups.has(template)) {
      const match = path.match(/^(.+\.)\d+(\..+)$/);
      templateGroups.set(template, {
        template,
        prefix: match![1],
        suffix: match![2],
        indices: [],
      });
    }
    templateGroups.get(template)!.indices.push(index);
  }

  // Find total count per template from all selectable paths
  const allTemplateCount = new Map<string, number>();
  for (const path of allSelectablePaths) {
    const parsed = toTemplate(path);
    if (parsed) {
      allTemplateCount.set(parsed.template, (allTemplateCount.get(parsed.template) ?? 0) + 1);
    }
  }

  const optimized: string[] = [...nonTemplated];

  for (const [template, group] of templateGroups) {
    const totalForTemplate = allTemplateCount.get(template) ?? 0;
    group.indices.sort((a, b) => a - b);

    if (group.indices.length === totalForTemplate) {
      // All layers selected for this pattern - use scoped wildcard
      const escapedPrefix = escapeRegex(group.prefix);
      const escapedSuffix = escapeRegex(group.suffix);
      optimized.push(`re:${escapedPrefix}\\d+${escapedSuffix}`);
    } else if (group.indices.length <= 3) {
      // Few indices - list explicitly
      for (const idx of group.indices) {
        optimized.push(`${group.prefix}${idx}${group.suffix}`);
      }
    } else {
      // Many but not all - use index-based regex
      // Check if it's a contiguous range
      const isContiguous = group.indices.every(
        (val, i) => i === 0 || val === group.indices[i - 1] + 1
      );

      if (isContiguous && group.indices.length > 3) {
        // Contiguous range
        const idxPattern = group.indices.map(String).join('|');
        const escapedPrefix = escapeRegex(group.prefix);
        const escapedSuffix = escapeRegex(group.suffix);
        optimized.push(`re:${escapedPrefix}(${idxPattern})${escapedSuffix}`);
      } else {
        const idxPattern = group.indices.map(String).join('|');
        const escapedPrefix = escapeRegex(group.prefix);
        const escapedSuffix = escapeRegex(group.suffix);
        optimized.push(`re:${escapedPrefix}(${idxPattern})${escapedSuffix}`);
      }
    }
  }

  return {
    explicit: [...selectedPaths],
    optimized,
  };
}
