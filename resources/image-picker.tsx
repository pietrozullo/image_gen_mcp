import { McpUseProvider, useWidget, type WidgetMetadata } from "mcp-use/react";
import React from "react";
import { z } from "zod";

import "./image-picker.css";

const pickerImageSchema = z.object({
  id: z.string(),
  prompt: z.string().optional(),
  dataUrl: z.string().optional(),
  path: z.string().optional(),
  url: z.string().optional(),
  downloadUrl: z.string().optional(),
  mimeType: z.string(),
  createdAt: z.string(),
});

const pickerPropsSchema = z.object({
  title: z.string().optional(),
  prompt: z.string().optional(),
  generatedAt: z.string().optional(),
  selectedId: z.string().optional(),
  images: z.array(pickerImageSchema),
});

export const widgetMetadata: WidgetMetadata = {
  description:
    "Image results picker for generated and transformed images with direct download actions.",
  props: pickerPropsSchema as any,
  exposeAsTool: false,
  metadata: {
    csp: {
      connectDomains: [],
      resourceDomains: [],
      scriptDirectives: ["'unsafe-eval'"],
    },
    prefersBorder: true,
    autoResize: true,
    widgetDescription:
      "Select an image result, preview it, and download it directly.",
  },
  annotations: {
    readOnlyHint: true,
  },
};

type PickerProps = z.infer<typeof pickerPropsSchema>;
type PickerImage = z.infer<typeof pickerImageSchema>;

type UnknownRecord = Record<string, unknown>;

const EMPTY_PROPS: PickerProps = {
  images: [],
};

function isObject(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null;
}

function normalizeProps(value: unknown): PickerProps {
  const parsed = pickerPropsSchema.safeParse(value);
  if (parsed.success) {
    return parsed.data;
  }
  return EMPTY_PROPS;
}

function formatDate(input: string): string {
  const parsed = new Date(input);
  return Number.isNaN(parsed.getTime()) ? input : parsed.toLocaleString();
}

function extensionFromMimeType(mimeType: string): string {
  switch (mimeType) {
    case "image/jpeg":
      return "jpg";
    case "image/gif":
      return "gif";
    case "image/webp":
      return "webp";
    case "image/bmp":
      return "bmp";
    case "image/tiff":
      return "tiff";
    case "image/png":
    default:
      return "png";
  }
}

function buildFilename(image: PickerImage, selectedIndex: number): string {
  return `generated-${selectedIndex}.${extensionFromMimeType(image.mimeType)}`;
}

function sanitizeBaseUrl(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

function resolveImageUrl(image: PickerImage, mcpUrl: string): string {
  const preferred = image.downloadUrl ?? image.url ?? image.path;
  if (!preferred) {
    return "";
  }

  if (
    preferred.startsWith("http://") ||
    preferred.startsWith("https://") ||
    preferred.startsWith("data:") ||
    preferred.startsWith("blob:")
  ) {
    return preferred;
  }

  const base = sanitizeBaseUrl(mcpUrl);
  if (!base) {
    return preferred;
  }

  if (preferred.startsWith("/")) {
    return `${base}${preferred}`;
  }

  return `${base}/${preferred}`;
}

function getRenderableSource(image: PickerImage, mcpUrl: string): string {
  return image.dataUrl ?? resolveImageUrl(image, mcpUrl);
}

const ImagePickerWidget: React.FC = () => {
  const { props, output, metadata, isPending, theme, mcp_url, openExternal } =
    useWidget<PickerProps, UnknownRecord, UnknownRecord>();

  const propsFromHook = normalizeProps(props);

  const propsFromMetadata = React.useMemo(() => {
    if (!isObject(metadata)) {
      return EMPTY_PROPS;
    }
    return normalizeProps(metadata["mcp-use/props"]);
  }, [metadata]);

  const propsFromOutput = React.useMemo(() => {
    if (!isObject(output)) {
      return EMPTY_PROPS;
    }

    const outputObject = output as UnknownRecord;
    if (isObject(outputObject.structuredContent)) {
      return normalizeProps(outputObject.structuredContent);
    }

    return normalizeProps(outputObject);
  }, [output]);

  const effectiveProps = React.useMemo(() => {
    const candidates = [propsFromHook, propsFromMetadata, propsFromOutput];

    const bestWithImages = candidates.find((candidate) => candidate.images.length > 0);
    const best = bestWithImages ?? candidates.find((candidate) => candidate.images.length >= 0) ?? EMPTY_PROPS;

    return {
      title: propsFromHook.title ?? propsFromMetadata.title ?? propsFromOutput.title,
      prompt: propsFromHook.prompt ?? propsFromMetadata.prompt ?? propsFromOutput.prompt,
      generatedAt:
        propsFromHook.generatedAt ??
        propsFromMetadata.generatedAt ??
        propsFromOutput.generatedAt,
      selectedId:
        propsFromHook.selectedId ?? propsFromMetadata.selectedId ?? propsFromOutput.selectedId,
      images: best.images,
    } satisfies PickerProps;
  }, [propsFromHook, propsFromMetadata, propsFromOutput]);

  const images = effectiveProps.images;
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [failedImages, setFailedImages] = React.useState<Record<string, string>>({});

  React.useEffect(() => {
    if (!images.length) {
      setSelectedId(null);
      return;
    }

    const preferredId =
      effectiveProps.selectedId &&
      images.some((image) => image.id === effectiveProps.selectedId)
        ? effectiveProps.selectedId
        : images[0].id;

    setSelectedId((current) => {
      if (current && images.some((image) => image.id === current)) {
        return current;
      }
      return preferredId;
    });
  }, [images, effectiveProps.selectedId]);

  const selectedImage = React.useMemo(() => {
    if (!images.length) {
      return null;
    }

    if (selectedId) {
      const match = images.find((image) => image.id === selectedId);
      if (match) {
        return match;
      }
    }

    return images[0];
  }, [images, selectedId]);

  const selectedIndex = selectedImage
    ? Math.max(
        1,
        images.findIndex((image) => image.id === selectedImage.id) + 1
      )
    : 1;

  const heading =
    effectiveProps.title ??
    `Image picker (${images.length} image${images.length === 1 ? "" : "s"})`;

  const handleImageError = React.useCallback(
    (image: PickerImage) => {
      const resolvedUrl = getRenderableSource(image, mcp_url);
      setFailedImages((current) => ({
        ...current,
        [image.id]: resolvedUrl || "(missing url)",
      }));
    },
    [mcp_url]
  );

  const handleImageLoad = React.useCallback((imageId: string) => {
    setFailedImages((current) => {
      if (!current[imageId]) {
        return current;
      }

      const next = { ...current };
      delete next[imageId];
      return next;
    });
  }, []);

  const handleDownload = React.useCallback(
    async (image: PickerImage, imageIndex: number) => {
      const renderableSource = getRenderableSource(image, mcp_url);
      const externalSource = resolveImageUrl(image, mcp_url);
      if (!renderableSource) {
        return;
      }

      try {
        const response = await fetch(renderableSource);
        if (!response.ok) {
          throw new Error(`Failed to download image (${response.status})`);
        }

        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = blobUrl;
        link.download = buildFilename(image, imageIndex);
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(blobUrl);
      } catch {
        if (externalSource) {
          openExternal(externalSource);
        }
      }
    },
    [mcp_url, openExternal]
  );

  const selectedImageUrl = selectedImage ? resolveImageUrl(selectedImage, mcp_url) : "";
  const selectedImageSource = selectedImage
    ? getRenderableSource(selectedImage, mcp_url)
    : "";
  const canOpenSelectedImage = selectedImageSource.length > 0;

  return (
    <McpUseProvider autoSize>
      <section
        className={`picker-shell ${theme === "dark" ? "theme-dark" : "theme-light"}`}
      >
        <header className="picker-header">
          <h2>{heading}</h2>
          <p className="picker-role">
            Role: generate and edit images, then let you pick and download the result.
          </p>
          {effectiveProps.prompt ? (
            <p className="picker-prompt">{effectiveProps.prompt}</p>
          ) : null}
          {effectiveProps.generatedAt ? (
            <p className="picker-timestamp">
              Updated: {formatDate(effectiveProps.generatedAt)}
            </p>
          ) : null}
        </header>

        {isPending ? (
          <div className="picker-loading" role="status" aria-live="polite">
            Generating images...
          </div>
        ) : null}

        {!isPending && !images.length ? (
          <div className="picker-empty">
            No images available yet. Run <code>generate_image_from_text</code> or{" "}
            <code>transform_image</code> to populate this picker.
          </div>
        ) : null}

        {!isPending && selectedImage ? (
          <div className="picker-panel">
            <aside className="picker-sidebar" aria-label="Image picker">
              {images.map((image, index) => {
                const isSelected = image.id === selectedImage.id;
                const imageUrl = getRenderableSource(image, mcp_url);
                return (
                  <button
                    key={image.id}
                    type="button"
                    className={`picker-thumb ${isSelected ? "is-selected" : ""}`}
                    onClick={() => setSelectedId(image.id)}
                  >
                    <img
                      src={imageUrl}
                      alt={`Generated image ${index + 1}`}
                      loading="lazy"
                      onError={() => handleImageError(image)}
                      onLoad={() => handleImageLoad(image.id)}
                    />
                    <span className="picker-thumb-label">Image {index + 1}</span>
                  </button>
                );
              })}
            </aside>

            <div className="picker-preview">
              <img
                className="picker-main-image"
                src={selectedImageSource}
                alt="Selected generated image"
                onError={() => handleImageError(selectedImage)}
                onLoad={() => handleImageLoad(selectedImage.id)}
              />

              {failedImages[selectedImage.id] ? (
                <p className="picker-error">
                  Unable to render image from: <code>{failedImages[selectedImage.id]}</code>
                </p>
              ) : null}

              <div className="picker-details">
                <p>
                  <strong>Selected:</strong> Image {selectedIndex}
                </p>
                <p>
                  <strong>Format:</strong> {selectedImage.mimeType}
                </p>
                <p>
                  <strong>Created:</strong> {formatDate(selectedImage.createdAt)}
                </p>
                <p className="picker-next-step">
                  Next: ask for another variation or open recent images to compare.
                </p>
              </div>

              <div className="picker-actions">
                <button
                  type="button"
                  className="picker-action picker-action-primary"
                  disabled={!canOpenSelectedImage}
                  onClick={() => handleDownload(selectedImage, selectedIndex)}
                >
                  Download Image
                </button>
                <button
                  type="button"
                  className="picker-action"
                  disabled={!canOpenSelectedImage}
                  onClick={() => {
                    if (!canOpenSelectedImage) {
                      return;
                    }
                    if (selectedImageUrl) {
                      openExternal(selectedImageUrl);
                    }
                  }}
                >
                  Open Full Size
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </McpUseProvider>
  );
};

export default ImagePickerWidget;
