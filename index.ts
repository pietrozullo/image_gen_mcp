import "dotenv/config";

import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { MCPServer, text, widget } from "mcp-use/server";
import { z } from "zod";

import {
  getImageGenerationPrompt,
  getImageTransformationPrompt,
} from "./prompts.js";

const DEFAULT_PORT = 3000;
const DEFAULT_TTL_SECONDS = 3600;
const DEFAULT_CLEANUP_INTERVAL_SECONDS = 300;
const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
const MAX_IMAGES_PER_REQUEST = 4;
const MAX_RECENT_IMAGES = 50;

const IMAGE_MODEL =
  process.env.GEMINI_IMAGE_MODEL ?? "gemini-3-pro-image-preview";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const SERVER_ICON_URL = "https://cdn.mcp-use.com/google.svg";

const SERVER_PORT = parseNumberEnv(process.env.PORT, DEFAULT_PORT);
const SERVER_HOST = process.env.HOST ?? "0.0.0.0";
const PUBLIC_BASE_URL =
  process.env.MCP_URL ?? `http://localhost:${SERVER_PORT}`;

const IMAGE_STORAGE_DIR = path.resolve(
  process.env.IMAGE_STORAGE_DIR ?? path.join(os.tmpdir(), "gemini_images")
);
const IMAGE_TTL_SECONDS = parseNumberEnv(
  process.env.IMAGE_TTL_SECONDS,
  DEFAULT_TTL_SECONDS
);
const CLEANUP_INTERVAL_SECONDS = parseNumberEnv(
  process.env.CLEANUP_INTERVAL_SECONDS,
  DEFAULT_CLEANUP_INTERVAL_SECONDS
);

const ALLOWED_IMAGE_MIME_TYPES = new Set<string>([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
  "image/bmp",
  "image/tiff",
]);

function normalizePublicBaseUrl(rawBaseUrl: string): string {
  const trimmed = rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;

  try {
    const parsed = new URL(trimmed);
    if (parsed.hostname === "0.0.0.0" || parsed.hostname === "::") {
      parsed.hostname = "localhost";
      return parsed.toString().replace(/\/$/, "");
    }
    return parsed.toString().replace(/\/$/, "");
  } catch {
    return trimmed;
  }
}

const normalizedBaseUrl = normalizePublicBaseUrl(PUBLIC_BASE_URL);

export const server = new MCPServer({
  name: "mcp-server-gemini-image-generator",
  title: "Gemini Image Generator",
  version: "1.0.0",
  description:
    "Generate and transform images with Gemini, with an image picker widget for preview and download.",
  icons: [
    {
      src: SERVER_ICON_URL,
      mimeType: "image/svg+xml",
      sizes: ["512x512", "256x256", "128x128"],
    },
  ],
  host: SERVER_HOST,
  baseUrl: normalizedBaseUrl,
});

type GeminiTextPart = { text: string };
type GeminiImagePart = {
  inlineData: {
    mimeType: string;
    data: string;
  };
};

type GeminiPart = GeminiTextPart | GeminiImagePart;

type StoredImage = {
  id: string;
  filePath: string;
  mimeType: string;
  createdAtMs: number;
};

type WidgetImage = {
  id: string;
  prompt: string;
  path: string;
  url: string;
  downloadUrl: string;
  dataUrl?: string;
  mimeType: string;
  createdAt: string;
};

const imageRegistry = new Map<string, StoredImage>();
let recentImages: WidgetImage[] = [];

function parseNumberEnv(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function normalizeMimeType(rawMimeType: string | null | undefined): string | null {
  if (!rawMimeType) {
    return null;
  }

  const normalized = rawMimeType.split(";")[0].trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  return normalized === "image/jpg" ? "image/jpeg" : normalized;
}

function extensionFromMimeType(mimeType: string): string {
  switch (mimeType) {
    case "image/png":
      return "png";
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
    default:
      return "png";
  }
}

function mimeTypeFromFilename(filename: string): string {
  const extension = path.extname(filename).toLowerCase();
  switch (extension) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    case ".bmp":
      return "image/bmp";
    case ".tif":
    case ".tiff":
      return "image/tiff";
    case ".png":
    default:
      return "image/png";
  }
}

function detectMimeType(imageBytes: Buffer): string | null {
  if (imageBytes.length >= 8) {
    const pngSignature =
      imageBytes[0] === 0x89 &&
      imageBytes[1] === 0x50 &&
      imageBytes[2] === 0x4e &&
      imageBytes[3] === 0x47 &&
      imageBytes[4] === 0x0d &&
      imageBytes[5] === 0x0a &&
      imageBytes[6] === 0x1a &&
      imageBytes[7] === 0x0a;

    if (pngSignature) {
      return "image/png";
    }
  }

  if (
    imageBytes.length >= 3 &&
    imageBytes[0] === 0xff &&
    imageBytes[1] === 0xd8 &&
    imageBytes[2] === 0xff
  ) {
    return "image/jpeg";
  }

  if (
    imageBytes.length >= 6 &&
    imageBytes.toString("ascii", 0, 6).startsWith("GIF8")
  ) {
    return "image/gif";
  }

  if (
    imageBytes.length >= 12 &&
    imageBytes.toString("ascii", 0, 4) === "RIFF" &&
    imageBytes.toString("ascii", 8, 12) === "WEBP"
  ) {
    return "image/webp";
  }

  if (imageBytes.length >= 2 && imageBytes.toString("ascii", 0, 2) === "BM") {
    return "image/bmp";
  }

  if (
    imageBytes.length >= 4 &&
    ((imageBytes[0] === 0x49 &&
      imageBytes[1] === 0x49 &&
      imageBytes[2] === 0x2a &&
      imageBytes[3] === 0x00) ||
      (imageBytes[0] === 0x4d &&
        imageBytes[1] === 0x4d &&
        imageBytes[2] === 0x00 &&
        imageBytes[3] === 0x2a))
  ) {
    return "image/tiff";
  }

  return null;
}

function validateImageBytes(
  imageBytes: Buffer,
  declaredMimeType?: string | null
): { bytes: Buffer; mimeType: string } {
  if (!imageBytes.length) {
    throw new Error("Image payload is empty");
  }

  if (imageBytes.length > MAX_IMAGE_SIZE_BYTES) {
    throw new Error(
      `Image exceeds maximum size of ${MAX_IMAGE_SIZE_BYTES} bytes`
    );
  }

  const normalizedDeclaredMimeType = normalizeMimeType(declaredMimeType);
  const detectedMimeType = detectMimeType(imageBytes);
  const mimeType = detectedMimeType ?? normalizedDeclaredMimeType;

  if (!mimeType || !ALLOWED_IMAGE_MIME_TYPES.has(mimeType)) {
    throw new Error(
      "Unsupported image format. Allowed: png, jpeg/jpg, gif, webp, bmp, tiff"
    );
  }

  return {
    bytes: imageBytes,
    mimeType,
  };
}

function getImagePath(filename: string): string {
  const safeFileName = path.basename(filename);
  return path.join(IMAGE_STORAGE_DIR, safeFileName);
}

function createImageUrl(filename: string): string {
  return `${normalizedBaseUrl}/images/${filename}`;
}

function createImagePath(filename: string): string {
  return `/images/${filename}`;
}

async function saveImage(
  imageBytes: Buffer,
  mimeType: string,
  prompt: string
): Promise<WidgetImage> {
  const id = randomUUID();
  const filename = `${id}.${extensionFromMimeType(mimeType)}`;
  const filePath = getImagePath(filename);
  const createdAtMs = Date.now();

  await fs.writeFile(filePath, imageBytes);

  imageRegistry.set(filename, {
    id,
    filePath,
    mimeType,
    createdAtMs,
  });

  const imageRecord: WidgetImage = {
    id,
    prompt,
    path: createImagePath(filename),
    url: createImageUrl(filename),
    downloadUrl: createImageUrl(filename),
    dataUrl: `data:${mimeType};base64,${imageBytes.toString("base64")}`,
    mimeType,
    createdAt: new Date(createdAtMs).toISOString(),
  };

  recentImages = [imageRecord, ...recentImages.filter((item) => item.id !== id)].slice(
    0,
    MAX_RECENT_IMAGES
  );

  return imageRecord;
}

async function cleanupOldImages(): Promise<void> {
  const now = Date.now();
  const ttlMs = IMAGE_TTL_SECONDS * 1000;

  for (const [filename, storedImage] of imageRegistry.entries()) {
    if (now - storedImage.createdAtMs <= ttlMs) {
      continue;
    }

    try {
      await fs.unlink(storedImage.filePath);
    } catch {
      // Ignore missing file race conditions.
    }

    imageRegistry.delete(filename);
    recentImages = recentImages.filter((item) => item.id !== storedImage.id);
  }
}

async function cleanupAllImages(): Promise<void> {
  for (const storedImage of imageRegistry.values()) {
    try {
      await fs.unlink(storedImage.filePath);
    } catch {
      // Ignore cleanup races during shutdown.
    }
  }

  imageRegistry.clear();
  recentImages = [];
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function buildImageSummary(
  action: "Generated" | "Transformed",
  prompt: string,
  images: WidgetImage[]
): string {
  const urls = images.map((image, index) => `${index + 1}. ${image.url}`).join("\n");
  return [
    "Role: I generate and edit images, then return direct download links.",
    `${action} ${images.length} image(s) for prompt: "${prompt}".`,
    "Next step: ask for another variation or call show_recent_images to compare results.",
    `Download URLs:\n${urls}`,
  ].join("\n\n");
}

function parseDataUrlImage(encodedImage: string): {
  bytes: Buffer;
  mimeType: string;
} {
  const match = encodedImage
    .trim()
    .match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([a-zA-Z0-9+/=\s]+)$/);

  if (!match) {
    throw new Error(
      "Invalid encoded_image format. Expected: data:image/[format];base64,[data]"
    );
  }

  const mimeType = normalizeMimeType(match[1]);
  const base64Data = match[2].replace(/\s+/g, "");
  const imageBytes = Buffer.from(base64Data, "base64");

  return validateImageBytes(imageBytes, mimeType);
}

async function loadImageFromUrl(imageUrl: string): Promise<{
  bytes: Buffer;
  mimeType: string;
}> {
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(imageUrl);
  } catch {
    throw new Error("image_url must be a valid URL");
  }

  if (!(parsedUrl.protocol === "https:" || parsedUrl.protocol === "http:")) {
    throw new Error("image_url must use http or https");
  }

  const response = await fetch(parsedUrl.toString());

  if (!response.ok) {
    throw new Error(`Failed to fetch image from URL (HTTP ${response.status})`);
  }

  const contentType = normalizeMimeType(response.headers.get("content-type"));
  const imageBytes = Buffer.from(await response.arrayBuffer());

  return validateImageBytes(imageBytes, contentType);
}

async function generateImageBytes(parts: GeminiPart[]): Promise<{
  bytes: Buffer;
  mimeType: string;
}> {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required");
  }

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
    IMAGE_MODEL
  )}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      contents: [
        {
          role: "user",
          parts,
        },
      ],
      generationConfig: {
        responseModalities: ["TEXT", "IMAGE"],
      },
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `Gemini API request failed (HTTP ${response.status}): ${errorBody.slice(0, 500)}`
    );
  }

  const payload = (await response.json()) as Record<string, unknown>;
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];

  for (const candidate of candidates) {
    const partsList =
      typeof candidate === "object" &&
      candidate !== null &&
      Array.isArray((candidate as any).content?.parts)
        ? (candidate as any).content.parts
        : [];

    for (const part of partsList) {
      const inlineData = (part as any)?.inlineData ?? (part as any)?.inline_data;
      const rawData = typeof inlineData?.data === "string" ? inlineData.data : null;
      if (!rawData) {
        continue;
      }

      const rawMimeType =
        normalizeMimeType(inlineData.mimeType ?? inlineData.mime_type) ??
        "image/png";
      const imageBytes = Buffer.from(rawData, "base64");
      return validateImageBytes(imageBytes, rawMimeType);
    }
  }

  throw new Error("Gemini API response did not include image data");
}

async function generateImageBatch(
  prompt: string,
  imageCount: number,
  parts: GeminiPart[]
): Promise<WidgetImage[]> {
  const images: WidgetImage[] = [];

  for (let index = 0; index < imageCount; index += 1) {
    const generatedImage = await generateImageBytes(parts);
    const savedImage = await saveImage(
      generatedImage.bytes,
      generatedImage.mimeType,
      prompt
    );
    images.push(savedImage);
  }

  return images;
}

async function hydrateRecentImage(image: WidgetImage): Promise<WidgetImage> {
  if (image.dataUrl) {
    return image;
  }

  const imageFilename = path.basename(
    image.path || (() => {
      try {
        return new URL(image.url).pathname;
      } catch {
        return image.url;
      }
    })()
  );

  if (!imageFilename) {
    return image;
  }

  try {
    const imageBytes = await fs.readFile(getImagePath(imageFilename));
    return {
      ...image,
      dataUrl: `data:${image.mimeType};base64,${imageBytes.toString("base64")}`,
    };
  } catch {
    return image;
  }
}

server.get("/images/:filename", async (c: any) => {
  const filename = c.req.param("filename");
  const storedImage = imageRegistry.get(filename);
  const filePath = storedImage?.filePath ?? getImagePath(filename);
  const mimeType = storedImage?.mimeType ?? mimeTypeFromFilename(filename);

  try {
    const imageBytes = await fs.readFile(filePath);
    return new Response(imageBytes, {
      status: 200,
      headers: {
        "content-type": mimeType,
        "cache-control": "no-store",
        "content-disposition": `inline; filename="${filename}"`,
      },
    });
  } catch {
    if (storedImage) {
      imageRegistry.delete(filename);
      recentImages = recentImages.filter((image) => image.id !== storedImage.id);
    }
    return c.text("Image not found", 404);
  }
});

server.get("/upload-info", (c: any) => {
  return c.json({
    endpoint: "/upload",
    method: "POST",
    usage: "curl -X POST -F 'image=@file.png' https://your-server/upload",
    maxSizeBytes: MAX_IMAGE_SIZE_BYTES,
    allowedFormats: Array.from(ALLOWED_IMAGE_MIME_TYPES),
    ttlSeconds: IMAGE_TTL_SECONDS,
  });
});

server.post("/upload", async (c: any) => {
  try {
    const body = await c.req.parseBody();
    const imageField = Array.isArray(body.image) ? body.image[0] : body.image;

    if (!imageField || typeof (imageField as any).arrayBuffer !== "function") {
      return c.json(
        {
          error: "No image file provided. Use multipart form field named 'image'.",
        },
        400
      );
    }

    const fileLike = imageField as {
      type?: string;
      arrayBuffer: () => Promise<ArrayBuffer>;
    };

    const imageBytes = Buffer.from(await fileLike.arrayBuffer());
    const validatedImage = validateImageBytes(imageBytes, fileLike.type);
    const uploadedImage = await saveImage(
      validatedImage.bytes,
      validatedImage.mimeType,
      "uploaded-source-image"
    );

    return c.json({
      url: uploadedImage.url,
      id: uploadedImage.id,
      mimeType: uploadedImage.mimeType,
      expiresInSeconds: IMAGE_TTL_SECONDS,
    });
  } catch (error) {
    return c.json(
      {
        error: toErrorMessage(error),
      },
      500
    );
  }
});

const generateImageSchema = z.object({
  prompt: z
    .string()
    .min(1)
    .describe("The prompt describing what should be generated"),
  image_count: z
    .number()
    .int()
    .min(1)
    .max(MAX_IMAGES_PER_REQUEST)
    .default(1)
    .describe("How many images to generate (1-4)"),
});

(server as any).tool(
  {
    name: "generate_image_from_text",
    description:
      "Create images from a text prompt and return an image picker UI with download actions.",
    schema: generateImageSchema,
    widget: {
      name: "image-picker",
      invoking: "Generating image results...",
      invoked: "Image results ready",
    },
  },
  async ({ prompt, image_count }: any) => {
    const generationPrompt = getImageGenerationPrompt(prompt);
    const images = await generateImageBatch(prompt, image_count, [
      { text: generationPrompt },
    ]);

    return widget({
      props: {
        title: `Generated ${images.length} image${images.length === 1 ? "" : "s"}`,
        prompt,
        generatedAt: new Date().toISOString(),
        images,
        selectedId: images[0]?.id,
      },
      output: text(buildImageSummary("Generated", prompt, images)),
    });
  }
);

const transformImageSchema = z.object({
  prompt: z
    .string()
    .min(1)
    .describe("The transformation instruction for the source image"),
  encoded_image: z
    .string()
    .optional()
    .describe(
      "Optional data URL image input (data:image/[format];base64,[data])"
    ),
  image_url: z
    .string()
    .url()
    .optional()
    .describe("Optional source image URL (takes precedence over encoded_image)"),
  image_count: z
    .number()
    .int()
    .min(1)
    .max(MAX_IMAGES_PER_REQUEST)
    .default(1)
    .describe("How many transformed images to return (1-4)"),
});

(server as any).tool(
  {
    name: "transform_image",
    description:
      "Edit an existing image using a prompt and return a picker UI with downloadable results.",
    schema: transformImageSchema,
    widget: {
      name: "image-picker",
      invoking: "Editing image results...",
      invoked: "Edited image results ready",
    },
  },
  async ({ prompt, encoded_image, image_url, image_count }: any) => {
    if (!image_url && !encoded_image) {
      throw new Error("Provide either image_url or encoded_image");
    }

    const sourceImage = image_url
      ? await loadImageFromUrl(image_url)
      : parseDataUrlImage(encoded_image as string);

    const transformationPrompt = getImageTransformationPrompt(prompt);
    const images = await generateImageBatch(prompt, image_count, [
      { text: transformationPrompt },
      {
        inlineData: {
          mimeType: sourceImage.mimeType,
          data: sourceImage.bytes.toString("base64"),
        },
      },
    ]);

    return widget({
      props: {
        title: `Transformed ${images.length} image${images.length === 1 ? "" : "s"}`,
        prompt,
        generatedAt: new Date().toISOString(),
        images,
        selectedId: images[0]?.id,
      },
      output: text(buildImageSummary("Transformed", prompt, images)),
    });
  }
);

const showRecentImagesSchema = z.object({
  limit: z
    .number()
    .int()
    .min(1)
    .max(30)
    .default(12)
    .describe("Maximum number of recent images to include"),
});

(server as any).tool(
  {
    name: "show_recent_images",
    description:
      "Reopen recent generated images in the picker UI for quick preview and download.",
    schema: showRecentImagesSchema,
    widget: {
      name: "image-picker",
      invoking: "Loading recent images...",
      invoked: "Recent images ready",
    },
  },
  async ({ limit }: any) => {
    const images = await Promise.all(
      recentImages.slice(0, limit).map((image) => hydrateRecentImage(image))
    );

    return widget({
      props: {
        title: "Recent generated images",
        generatedAt: new Date().toISOString(),
        images,
        selectedId: images[0]?.id,
      },
      output: text(
        images.length
          ? `Showing ${images.length} recent generated image(s).`
          : "No generated images are currently available in memory."
      ),
    });
  }
);

let hasStarted = false;

export async function startServer(): Promise<void> {
  if (hasStarted) {
    return;
  }
  hasStarted = true;

  await fs.mkdir(IMAGE_STORAGE_DIR, { recursive: true });

  const cleanupTimer = setInterval(() => {
    void cleanupOldImages();
  }, CLEANUP_INTERVAL_SECONDS * 1000);
  const timerWithUnref = cleanupTimer as unknown as { unref?: () => void };
  if (typeof timerWithUnref.unref === "function") {
    timerWithUnref.unref();
  }

  const shutdown = async () => {
    await cleanupAllImages();
    process.exit(0);
  };

  process.once("SIGINT", () => {
    void shutdown();
  });

  process.once("SIGTERM", () => {
    void shutdown();
  });

  await server.listen(SERVER_PORT);

  console.log(`Gemini Image Generator MCP server running on ${normalizedBaseUrl}`);
  console.log(`MCP endpoint: ${normalizedBaseUrl}/mcp`);
  console.log(`Image upload endpoint: ${normalizedBaseUrl}/upload`);
}

try {
  await startServer();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Failed to start server: ${message}`);
  process.exit(1);
}
