import base64
import os
import logging
import sys
from io import BytesIO
from typing import List, Tuple

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp_use.server import MCPServer
from mcp.server.fastmcp import Image

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

# Initialize MCP server with cloud-friendly defaults (0.0.0.0 disables DNS rebinding protection)
server = MCPServer(
    name="mcp-server-gemini-image-generator",
    version="0.1.0",
    instructions="Generate and transform images using Google's Gemini AI model",
)


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


# ==================== MCP Tools ====================

@server.tool()
async def generate_image_from_text(prompt: str) -> Image:
    """Generate an image based on the given text prompt using Google's Gemini model.

    Args:
        prompt: User's text prompt describing the desired image to generate

    Returns:
        Generated image
    """
    logger.info(f"Generating image from prompt: {prompt}")

    contents = get_image_generation_prompt(prompt)
    image_bytes = await generate_image_bytes([contents])

    logger.info("Image generated successfully")
    return Image(data=image_bytes, format="png")


@server.tool()
async def transform_image(encoded_image: str, prompt: str) -> Image:
    """Transform an existing image based on the given text prompt using Google's Gemini model.

    Args:
        encoded_image: Base64 encoded image data with header. Must be in format:
                    "data:image/[format];base64,[data]"
                    Where [format] can be: png, jpeg, jpg, gif, webp, etc.
        prompt: Text prompt describing the desired transformation or modifications

    Returns:
        Transformed image
    """
    logger.info(f"Transforming image with prompt: {prompt}")

    source_image, _ = load_image_from_base64(encoded_image)
    edit_instructions = get_image_transformation_prompt(prompt)
    image_bytes = await generate_image_bytes([edit_instructions, source_image])

    logger.info("Image transformed successfully")
    return Image(data=image_bytes, format="png")


if __name__ == "__main__":
    logger.info("Starting Gemini Image Generator MCP server...")
    server.run(transport="streamable-http", host="0.0.0.0", port=3000)
