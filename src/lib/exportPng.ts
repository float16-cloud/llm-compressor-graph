import { toPng } from 'html-to-image';

export interface ExportPngOptions {
  /** The DOM element to capture */
  element: HTMLElement;
  /** Filename for the download (without extension) */
  filename?: string;
  /** Background color to use (should match the app's canvas color) */
  backgroundColor?: string;
  /** Pixel ratio for high-DPI exports */
  pixelRatio?: number;
  /** Model name to embed as header in the image */
  modelName?: string;
}

export async function exportTreeAsPng({
  element,
  filename = 'architecture-tree',
  backgroundColor = '#ffffff',
  pixelRatio = 2,
  modelName,
}: ExportPngOptions): Promise<void> {
  // 1. Save original styles
  const originalMaxHeight = element.style.maxHeight;
  const originalOverflow = element.style.overflow;

  // 2. Create header element if model name is provided
  let headerElement: HTMLDivElement | null = null;
  if (modelName) {
    headerElement = document.createElement('div');
    headerElement.style.cssText = `
      padding: 16px 12px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
      font-family: 'Bricolage Grotesque', system-ui, sans-serif;
      font-size: 15px;
      font-weight: 600;
      color: #1f2937;
    `;
    headerElement.textContent = modelName;
    element.insertBefore(headerElement, element.firstChild);
  }

  try {
    // 3. Remove scroll constraints so full tree is visible
    element.style.maxHeight = 'none';
    element.style.overflow = 'visible';

    // 4. Capture to PNG data URL
    //    html-to-image may need multiple passes for web fonts to load.
    //    We call toPng twice: first pass warms the font cache, second
    //    pass produces the correct output. This is a known workaround.
    //    See: https://github.com/bubkoo/html-to-image/issues/361
    await toPng(element, { backgroundColor, pixelRatio, skipAutoScale: true });
    const dataUrl = await toPng(element, { backgroundColor, pixelRatio, skipAutoScale: true });

    // 5. Trigger download
    const link = document.createElement('a');
    link.download = `${filename}.png`;
    link.href = dataUrl;
    link.click();
  } finally {
    // 6. Remove header element if it was added
    if (headerElement && element.contains(headerElement)) {
      element.removeChild(headerElement);
    }

    // 7. Restore original styles (always, even on error)
    element.style.maxHeight = originalMaxHeight;
    element.style.overflow = originalOverflow;
  }
}
