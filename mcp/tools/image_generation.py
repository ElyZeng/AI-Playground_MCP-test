"""
Image Generation Tools for AI Playground MCP Server
"""

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin

import httpx
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from PIL import Image


class ImageGenerationTools:
    """Tools for AI Playground image generation functionality"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for image generation
        
    async def get_tools(self) -> List[Tool]:
        """Get all available image generation tools"""
        return [
            Tool(
                name="generate_image",
                description="Generate an image from text prompt using AI Playground",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt describing the image to generate"
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Negative prompt to avoid certain elements",
                            "default": ""
                        },
                        "width": {
                            "type": "integer",
                            "description": "Image width in pixels",
                            "default": 512,
                            "minimum": 64,
                            "maximum": 2048
                        },
                        "height": {
                            "type": "integer", 
                            "description": "Image height in pixels",
                            "default": 512,
                            "minimum": 64,
                            "maximum": 2048
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Number of inference steps",
                            "default": 20,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "cfg_scale": {
                            "type": "number",
                            "description": "Classifier Free Guidance scale",
                            "default": 7.0,
                            "minimum": 1.0,
                            "maximum": 20.0
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility (-1 for random)",
                            "default": -1
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Model to use for generation",
                            "default": ""
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="image_to_image", 
                description="Transform an existing image based on a text prompt",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt describing the desired transformation"
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Path to the input image file"
                        },
                        "strength": {
                            "type": "number",
                            "description": "Transformation strength (0.0-1.0)",
                            "default": 0.8,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "negative_prompt": {
                            "type": "string", 
                            "description": "Negative prompt",
                            "default": ""
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Number of inference steps", 
                            "default": 20
                        },
                        "cfg_scale": {
                            "type": "number",
                            "description": "CFG scale",
                            "default": 7.0
                        }
                    },
                    "required": ["prompt", "image_path"]
                }
            ),
            Tool(
                name="upscale_image",
                description="Upscale an image using AI enhancement",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image to upscale"
                        },
                        "scale_factor": {
                            "type": "number",
                            "description": "Upscaling factor (1.5-4.0)",
                            "default": 2.0,
                            "minimum": 1.5,
                            "maximum": 4.0
                        },
                        "model_type": {
                            "type": "string", 
                            "description": "Upscaling model type",
                            "enum": ["esrgan", "real_esrgan", "swinir"],
                            "default": "real_esrgan"
                        }
                    },
                    "required": ["image_path"]
                }
            )
        ]
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Execute image generation tool"""
        
        if name == "generate_image":
            return await self._generate_image(arguments)
        elif name == "image_to_image":
            return await self._image_to_image(arguments)
        elif name == "upscale_image":
            return await self._upscale_image(arguments)
        else:
            raise ValueError(f"Unknown image tool: {name}")
    
    async def _generate_image(self, args: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
        """Generate image from text prompt"""
        try:
            # Prepare request payload
            payload = {
                "prompt": args["prompt"],
                "negative_prompt": args.get("negative_prompt", ""),
                "width": args.get("width", 512),
                "height": args.get("height", 512), 
                "steps": args.get("steps", 20),
                "cfg_scale": args.get("cfg_scale", 7.0),
                "seed": args.get("seed", -1)
            }
            
            if args.get("model_name"):
                payload["model_name"] = args["model_name"]
            
            # Send request to AI Playground backend
            url = urljoin(self.base_url, "/api/image/generate")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                image_data = result["data"]
                
                # Handle base64 encoded image
                if "image" in image_data:
                    image_b64 = image_data["image"]
                    if image_b64.startswith("data:image"):
                        # Remove data URL prefix
                        image_b64 = image_b64.split(",", 1)[1]
                    
                    return [
                        TextContent(
                            type="text", 
                            text=f"Generated image with prompt: '{args['prompt']}'\nSeed: {image_data.get('seed', 'unknown')}"
                        ),
                        ImageContent(
                            type="image",
                            data=image_b64,
                            mimeType="image/png"
                        )
                    ]
                else:
                    return [TextContent(type="text", text="Image generated successfully but no image data returned")]
            else:
                error_msg = result.get("message", "Unknown error occurred")
                return [TextContent(type="text", text=f"Image generation failed: {error_msg}")]
                
        except httpx.RequestError as e:
            return [TextContent(type="text", text=f"Network error: {str(e)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating image: {str(e)}")]
    
    async def _image_to_image(self, args: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
        """Transform image based on prompt"""
        try:
            image_path = Path(args["image_path"])
            if not image_path.exists():
                return [TextContent(type="text", text=f"Input image not found: {image_path}")]
            
            # Read and encode input image
            with open(image_path, "rb") as f:
                input_image_b64 = base64.b64encode(f.read()).decode()
            
            payload = {
                "prompt": args["prompt"],
                "image": f"data:image/png;base64,{input_image_b64}",
                "strength": args.get("strength", 0.8),
                "negative_prompt": args.get("negative_prompt", ""),
                "steps": args.get("steps", 20),
                "cfg_scale": args.get("cfg_scale", 7.0)
            }
            
            url = urljoin(self.base_url, "/api/image/img2img")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data", {}).get("image"):
                image_b64 = result["data"]["image"]
                if image_b64.startswith("data:image"):
                    image_b64 = image_b64.split(",", 1)[1]
                
                return [
                    TextContent(
                        type="text",
                        text=f"Transformed image with prompt: '{args['prompt']}'\nStrength: {args.get('strength', 0.8)}"
                    ),
                    ImageContent(
                        type="image",
                        data=image_b64,
                        mimeType="image/png"
                    )
                ]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Image transformation failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error in image-to-image: {str(e)}")]
    
    async def _upscale_image(self, args: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
        """Upscale image using AI enhancement"""
        try:
            image_path = Path(args["image_path"])
            if not image_path.exists():
                return [TextContent(type="text", text=f"Input image not found: {image_path}")]
            
            with open(image_path, "rb") as f:
                input_image_b64 = base64.b64encode(f.read()).decode()
            
            payload = {
                "image": f"data:image/png;base64,{input_image_b64}",
                "scale_factor": args.get("scale_factor", 2.0),
                "model_type": args.get("model_type", "real_esrgan")
            }
            
            url = urljoin(self.base_url, "/api/image/upscale")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data", {}).get("image"):
                image_b64 = result["data"]["image"]
                if image_b64.startswith("data:image"):
                    image_b64 = image_b64.split(",", 1)[1]
                
                return [
                    TextContent(
                        type="text",
                        text=f"Upscaled image by {args.get('scale_factor', 2.0)}x using {args.get('model_type', 'real_esrgan')}"
                    ),
                    ImageContent(
                        type="image",
                        data=image_b64,
                        mimeType="image/png"
                    )
                ]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Image upscaling failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error upscaling image: {str(e)}")]
    
    def __del__(self):
        """Cleanup HTTP client"""
        if hasattr(self, 'client'):
            asyncio.create_task(self.client.aclose())