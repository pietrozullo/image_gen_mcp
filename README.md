# Gemini Image Generator MCP Server (TypeScript)

TypeScript `mcp-use` MCP server that generates and transforms images with Gemini, then renders them in a UI image picker where users can preview and download each image.

Built for remote hosting with streamable HTTP (`/mcp`).

## What This Includes

- `generate_image_from_text` tool (single or multiple images)
- `transform_image` tool (image-to-image editing)
- `show_recent_images` tool (reopen recent gallery)
- React widget `image-picker` in `resources/`:
  - Select between generated images
  - Large preview
  - Download selected image from the UI
- HTTP routes:
  - `GET /images/:filename` (serves generated files)
  - `POST /upload` (upload source image, get URL back)
  - `GET /upload-info`

## Streamable HTTP Endpoint

When running, connect clients to:

```text
https://your-domain.com/mcp
```

`mcp-use` serves streamable HTTP at `/mcp` and also mounts `/inspector` for debugging.

## Requirements

- Node.js 20+
- Gemini API key

## Environment Variables

| Variable | Required | Description | Default |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key | - |
| `GEMINI_IMAGE_MODEL` | No | Gemini image model id | `gemini-3-pro-image-preview` |
| `PORT` | No | Server port | `3000` |
| `HOST` | No | Bind host | `0.0.0.0` |
| `MCP_URL` | No | Public base URL used for image download links | `http://localhost:3000` |
| `IMAGE_STORAGE_DIR` | No | Directory for generated images | OS temp dir + `gemini_images` |
| `IMAGE_TTL_SECONDS` | No | Image retention in seconds | `3600` |
| `CLEANUP_INTERVAL_SECONDS` | No | Cleanup interval in seconds | `300` |

## Local Development

```bash
npm install
npm run dev
```

Useful URLs:

- MCP endpoint: `http://localhost:3000/mcp`
- Inspector: `http://localhost:3000/inspector`
- Upload info: `http://localhost:3000/upload-info`

## Production

```bash
npm install
npm run build
npm run start
```

For remote deployment, set:

```bash
export MCP_URL="https://your-public-domain.com"
```

## Tool Inputs

### `generate_image_from_text`

- `prompt` (string, required)
- `image_count` (number, optional, 1-4, default `1`)

### `transform_image`

- `prompt` (string, required)
- `image_url` (string URL, optional)
- `encoded_image` (base64 data URL, optional)
- `image_count` (number, optional, 1-4, default `1`)

`image_url` takes precedence when both image inputs are provided.

### `show_recent_images`

- `limit` (number, optional, 1-30, default `12`)

## Docker

```bash
docker build -t gemini-image-mcp .
docker run -p 3000:3000 \
  -e GEMINI_API_KEY="your-key" \
  -e MCP_URL="https://your-domain.com" \
  gemini-image-mcp
```
