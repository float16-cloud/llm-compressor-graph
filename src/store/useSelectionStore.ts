import { create } from 'zustand';

interface SelectionState {
  selectedPaths: Set<string>;
  allSelectablePaths: string[];
  paramCountMap: Map<string, number>;

  setAllSelectablePaths: (paths: string[]) => void;
  setParamCountMap: (map: Map<string, number>) => void;
  setSelectedPaths: (paths: string[]) => void;
  toggle: (path: string) => void;
  selectPaths: (paths: string[]) => void;
  deselectPaths: (paths: string[]) => void;
  selectAll: () => void;
  deselectAll: () => void;
  invertSelection: () => void;
  selectByPattern: (pattern: RegExp) => void;
  deselectByPattern: (pattern: RegExp) => void;
  selectRange: (startLayer: number, endLayer: number, suffixPattern?: RegExp) => void;
  isSelected: (path: string) => boolean;
}

export const useSelectionStore = create<SelectionState>((set, get) => ({
  selectedPaths: new Set<string>(),
  allSelectablePaths: [],
  paramCountMap: new Map<string, number>(),

  setAllSelectablePaths: (paths) => set({ allSelectablePaths: paths, selectedPaths: new Set() }),

  setParamCountMap: (map) => set({ paramCountMap: map }),

  setSelectedPaths: (paths) => set((state) => {
    const valid = new Set(state.allSelectablePaths);
    const next = new Set<string>();
    for (const p of paths) {
      if (valid.has(p)) next.add(p);
    }
    return { selectedPaths: next };
  }),

  toggle: (path) => set((state) => {
    const next = new Set(state.selectedPaths);
    if (next.has(path)) {
      next.delete(path);
    } else {
      next.add(path);
    }
    return { selectedPaths: next };
  }),

  selectPaths: (paths) => set((state) => {
    const next = new Set(state.selectedPaths);
    for (const p of paths) next.add(p);
    return { selectedPaths: next };
  }),

  deselectPaths: (paths) => set((state) => {
    const next = new Set(state.selectedPaths);
    for (const p of paths) next.delete(p);
    return { selectedPaths: next };
  }),

  selectAll: () => set((state) => ({
    selectedPaths: new Set(state.allSelectablePaths),
  })),

  deselectAll: () => set({ selectedPaths: new Set() }),

  invertSelection: () => set((state) => {
    const next = new Set<string>();
    for (const p of state.allSelectablePaths) {
      if (!state.selectedPaths.has(p)) {
        next.add(p);
      }
    }
    return { selectedPaths: next };
  }),

  selectByPattern: (pattern) => set((state) => {
    const next = new Set(state.selectedPaths);
    for (const p of state.allSelectablePaths) {
      if (pattern.test(p)) next.add(p);
    }
    return { selectedPaths: next };
  }),

  deselectByPattern: (pattern) => set((state) => {
    const next = new Set(state.selectedPaths);
    for (const p of state.allSelectablePaths) {
      if (pattern.test(p)) next.delete(p);
    }
    return { selectedPaths: next };
  }),

  selectRange: (startLayer, endLayer, suffixPattern) => set((state) => {
    const next = new Set(state.selectedPaths);
    const layerPattern = /\.(\d+)\./;
    for (const p of state.allSelectablePaths) {
      const match = p.match(layerPattern);
      if (match) {
        const idx = parseInt(match[1], 10);
        if (idx >= startLayer && idx <= endLayer) {
          if (!suffixPattern || suffixPattern.test(p)) {
            next.add(p);
          }
        }
      }
    }
    return { selectedPaths: next };
  }),

  isSelected: (path) => get().selectedPaths.has(path),
}));
