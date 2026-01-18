import base64
import os
import logging
import sys
import uuid
import tempfile
import time
import threading
import atexit
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import httpx
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp_use.server import MCPServer
from mcp.server.fastmcp import Image
from starlette.responses import FileResponse, Response, JSONResponse

from prompts import get_image_generation_prompt, get_image_transformation_prompt

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Image generation model (nano banana pro)
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

# Image storage configuration
IMAGE_STORAGE_DIR = Path(os.environ.get("IMAGE_STORAGE_DIR", tempfile.gettempdir())) / "gemini_images"
IMAGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SERVER_PORT = int(os.environ.get("PORT", "3000"))
# Public URL base for accessing images
PUBLIC_URL_BASE = os.environ.get("MCP_URL", f"http://localhost:{SERVER_PORT}")

# Image cleanup configuration
IMAGE_TTL_SECONDS = int(os.environ.get("IMAGE_TTL_SECONDS", "3600"))  # Default: 1 hour
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", "300"))  # Default: 5 minutes

# Image validation configuration
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB max
ALLOWED_IMAGE_FORMATS = {'PNG', 'JPEG', 'GIF', 'WEBP', 'BMP', 'TIFF'}
ALLOWED_IMAGE_MIME_TYPES = {
    'image/png', 'image/jpeg', 'image/jpg', 'image/gif',
    'image/webp', 'image/bmp', 'image/tiff'
}

# Track created images with their creation time
_image_registry: dict[str, float] = {}
_registry_lock = threading.Lock()


def _cleanup_old_images():
    """Remove images older than IMAGE_TTL_SECONDS."""
    current_time = time.time()
    to_remove = []

    with _registry_lock:
        for filename, created_at in list(_image_registry.items()):
            if current_time - created_at > IMAGE_TTL_SECONDS:
                filepath = IMAGE_STORAGE_DIR / filename
                try:
                    if filepath.exists():
                        filepath.unlink()
                        logger.info(f"Cleaned up expired image: {filename}")
                except Exception as e:
                    logger.error(f"Failed to cleanup image {filename}: {e}")
                to_remove.append(filename)

        for filename in to_remove:
            del _image_registry[filename]


def _cleanup_loop():
    """Background thread that periodically cleans up old images."""
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            _cleanup_old_images()
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")


def _cleanup_all_images():
    """Remove all tracked images on shutdown."""
    logger.info("Shutting down: cleaning up all session images...")
    with _registry_lock:
        for filename in list(_image_registry.keys()):
            filepath = IMAGE_STORAGE_DIR / filename
            try:
                if filepath.exists():
                    filepath.unlink()
                    logger.info(f"Shutdown cleanup: removed {filename}")
            except Exception as e:
                logger.error(f"Shutdown cleanup failed for {filename}: {e}")
        _image_registry.clear()


# Start cleanup thread and register shutdown handler
_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()
atexit.register(_cleanup_all_images)

# Initialize MCP server with cloud-friendly defaults (0.0.0.0 disables DNS rebinding protection)
server = MCPServer(
    name="mcp-server-gemini-image-generator",
    version="0.1.0",
    instructions="Generate and transform images using Google's Gemini AI model",
    host="0.0.0.0",
    port=SERVER_PORT,
)


# ==================== Image File Serving ====================

@server.custom_route("/images/{filename}", methods=["GET"])
async def serve_image(request):
    """Serve generated images from the storage directory."""
    filename = request.path_params["filename"]
    filepath = IMAGE_STORAGE_DIR / filename

    if not filepath.exists() or not filepath.is_file():
        return Response("Image not found", status_code=404)

    # Security: ensure we're serving from the expected directory
    if not filepath.resolve().is_relative_to(IMAGE_STORAGE_DIR.resolve()):
        return Response("Access denied", status_code=403)

    return FileResponse(filepath, media_type="image/png")


@server.custom_route("/upload-info", methods=["GET"])
async def upload_info(request):
    """Return upload endpoint info."""
    return JSONResponse({
        "endpoint": "/upload",
        "method": "POST",
        "usage": "curl -X POST -F 'image=@file.jpg' URL/upload",
        "max_size_bytes": MAX_IMAGE_SIZE_BYTES,
        "allowed_formats": list(ALLOWED_IMAGE_FORMATS),
        "ttl_seconds": IMAGE_TTL_SECONDS
    })


@server.custom_route("/upload", methods=["POST"])
async def upload_image(request):
    """Upload an image and get a URL back for use with transform_image.

    Accepts multipart form data with an 'image' field.
    Returns JSON with the image URL.

    Usage:
        curl -X POST -F "image=@/path/to/image.jpg" https://server/upload
    """
    try:
        form = await request.form()
        image_file = form.get("image")

        if not image_file:
            return JSONResponse(
                {"error": "No image file provided. Use: curl -X POST -F 'image=@file.jpg' URL"},
                status_code=400
            )

        # Read the uploaded file
        image_bytes = await image_file.read()

        # Check size limit
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            return JSONResponse(
                {"error": f"Image too large: {len(image_bytes)} bytes (max: {MAX_IMAGE_SIZE_BYTES})"},
                status_code=400
            )

        # Validate it's actually an image (not an executable or other file)
        try:
            image = PIL.Image.open(BytesIO(image_bytes))
            image.load()  # Force load to detect corrupted images

            pil_format = image.format
            if pil_format is None:
                return JSONResponse({"error": "Could not determine image format"}, status_code=400)

            if pil_format.upper() not in ALLOWED_IMAGE_FORMATS:
                return JSONResponse(
                    {"error": f"Unsupported image format: {pil_format}. Allowed: {ALLOWED_IMAGE_FORMATS}"},
                    status_code=400
                )
        except PIL.UnidentifiedImageError:
            return JSONResponse({"error": "File is not a valid image"}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": f"Image validation failed: {e}"}, status_code=400)

        # Save the image
        filename = f"{uuid.uuid4()}.png"
        filepath = IMAGE_STORAGE_DIR / filename

        # Convert to PNG for consistency
        output = BytesIO()
        image.save(output, format="PNG")
        filepath.write_bytes(output.getvalue())

        # Register for cleanup
        with _registry_lock:
            _image_registry[filename] = time.time()

        image_url = f"{PUBLIC_URL_BASE}/images/{filename}"
        logger.info(f"Uploaded image saved to {filepath} (TTL: {IMAGE_TTL_SECONDS}s)")

        return JSONResponse({
            "url": image_url,
            "filename": filename,
            "format": pil_format,
            "size": image.size,
            "expires_in_seconds": IMAGE_TTL_SECONDS
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def save_image_to_disk(image_bytes: bytes) -> str:
    """Save image bytes to disk and return the download URL.

    Args:
        image_bytes: Raw image bytes to save

    Returns:
        Public URL to download the image
    """
    filename = f"{uuid.uuid4()}.png"
    filepath = IMAGE_STORAGE_DIR / filename
    filepath.write_bytes(image_bytes)

    # Register for cleanup
    with _registry_lock:
        _image_registry[filename] = time.time()

    logger.info(f"Saved image to {filepath} (TTL: {IMAGE_TTL_SECONDS}s)")
    return f"{PUBLIC_URL_BASE}/images/{filename}"


# ==================== Gemini API Interaction ====================

async def generate_image_bytes(contents: List) -> bytes:
    """Generate an image with Gemini and return raw bytes.

    Args:
        contents: List containing the prompt and optionally an image

    Returns:
        Raw image bytes
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image']
        )
    )

    logger.info(f"Response received from Gemini API using model {IMAGE_MODEL}")

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise ValueError("No image data found in Gemini response")


# ==================== Image Processing Functions ====================

def load_image_from_base64(encoded_image: str) -> Tuple[PIL.Image.Image, str]:
    """Load an image from a base64-encoded string.

    Args:
        encoded_image: Base64 encoded image data with header

    Returns:
        Tuple containing the PIL Image object and the image format
    """
    if not encoded_image.startswith('data:image/'):
        raise ValueError("Invalid image format. Expected data:image/[format];base64,[data]")

    image_format, image_data = encoded_image.split(';base64,')
    image_format = image_format.replace('data:', '')
    image_bytes = base64.b64decode(image_data)
    source_image = PIL.Image.open(BytesIO(image_bytes))
    logger.info(f"Successfully loaded image with format: {image_format}")
    return source_image, image_format


def validate_image_bytes(image_bytes: bytes) -> PIL.Image.Image:
    """Validate that bytes represent a valid image file.

    Args:
        image_bytes: Raw bytes to validate

    Returns:
        PIL Image object if valid

    Raises:
        ValueError: If the bytes are not a valid image
    """
    try:
        # Attempt to open with PIL - this validates it's a real image
        image = PIL.Image.open(BytesIO(image_bytes))
        # Force load to detect truncated/corrupted images
        image.load()

        # Check the format is an allowed image type
        pil_format = image.format
        if pil_format is None:
            raise ValueError("Could not determine image format")

        if pil_format.upper() not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {pil_format}")

        logger.info(f"Validated image: format={pil_format}, size={image.size}")
        return image

    except PIL.UnidentifiedImageError:
        raise ValueError("File is not a valid image")
    except Exception as e:
        raise ValueError(f"Image validation failed: {e}")


async def load_image_from_url(image_url: str) -> PIL.Image.Image:
    """Fetch and validate an image from a URL.

    Args:
        image_url: URL to fetch the image from

    Returns:
        PIL Image object

    Raises:
        ValueError: If the URL is invalid, unreachable, or doesn't contain a valid image
    """
    logger.info(f"Fetching image from URL: {image_url}")

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # First, do a HEAD request to check content-type and size
            head_response = await client.head(image_url)
            head_response.raise_for_status()

            content_type = head_response.headers.get('content-type', '').split(';')[0].strip().lower()
            content_length = head_response.headers.get('content-length')

            # Validate content type
            if content_type and content_type not in ALLOWED_IMAGE_MIME_TYPES:
                raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")

            # Check size if provided
            if content_length:
                size = int(content_length)
                if size > MAX_IMAGE_SIZE_BYTES:
                    raise ValueError(f"Image too large: {size} bytes (max: {MAX_IMAGE_SIZE_BYTES})")

            # Now fetch the actual content
            response = await client.get(image_url)
            response.raise_for_status()

            image_bytes = response.content

            # Final size check
            if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Image too large: {len(image_bytes)} bytes (max: {MAX_IMAGE_SIZE_BYTES})")

            # Validate it's actually an image
            image = validate_image_bytes(image_bytes)

            logger.info(f"Successfully loaded image from URL: {image_url}")
            return image

    except httpx.HTTPStatusError as e:
        raise ValueError(f"Failed to fetch image: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise ValueError(f"Failed to fetch image: {e}")


# ==================== MCP Tools ====================

@server.tool()
async def generate_image_from_text(prompt: str) -> list:
    """Generate an image based on the given text prompt using Google's Gemini model.

    Args:
        prompt: User's text prompt describing the desired image to generate

    Returns:
        Generated image and download URL for programmatic use
    """
    logger.info(f"Generating image from prompt: {prompt}")

    contents = get_image_generation_prompt(prompt)
    image_bytes = await generate_image_bytes([contents])
    download_url = save_image_to_disk(image_bytes)

    logger.info(f"Image generated successfully, available at: {download_url}")
    return [
        Image(data=image_bytes, format="png"),
        f"Download URL: {download_url}"
    ]


@server.tool()
async def transform_image(
    prompt: str,
    encoded_image: Optional[str] = None,
    image_url: Optional[str] = None
) -> list:
    """Transform an existing image based on the given text prompt using Google's Gemini model.

    Args:
        prompt: Text prompt describing the desired transformation or modifications
        encoded_image: Base64 encoded image data with header. Must be in format:
                    "data:image/[format];base64,[data]"
                    Where [format] can be: png, jpeg, jpg, gif, webp, etc.
        image_url: URL to fetch the source image from. Use this instead of encoded_image
                   to avoid passing large base64 strings. The URL must point to a valid
                   image file (PNG, JPEG, GIF, WebP, BMP, or TIFF). Max size: 10MB.

    Returns:
        Transformed image and download URL for programmatic use

    Note:
        Provide either encoded_image OR image_url, not both. If both are provided,
        image_url takes precedence.
    """
    logger.info(f"Transforming image with prompt: {prompt}")

    # Determine source image
    if image_url:
        source_image = await load_image_from_url(image_url)
    elif encoded_image:
        source_image, _ = load_image_from_base64(encoded_image)
    else:
        raise ValueError("Either encoded_image or image_url must be provided")

    edit_instructions = get_image_transformation_prompt(prompt)
    image_bytes = await generate_image_bytes([edit_instructions, source_image])
    download_url = save_image_to_disk(image_bytes)

    logger.info(f"Image transformed successfully, available at: {download_url}")
    return [
        Image(data=image_bytes, format="png"),
        f"Download URL: {download_url}"
    ]


if __name__ == "__main__":
    logger.info("Starting Gemini Image Generator MCP server...")
    logger.info(f"Images will be served at {PUBLIC_URL_BASE}/images/")
    server.run(transport="streamable-http", host="0.0.0.0", port=SERVER_PORT)
