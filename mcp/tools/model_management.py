"""
Model Management Tools for AI Playground MCP Server
"""

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin

import httpx
from mcp.types import Tool, TextContent


class ModelManagementTools:
    """Tools for AI Playground model management functionality"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=600.0)  # 10 minute timeout for model operations
        
    async def get_tools(self) -> List[Tool]:
        """Get all available model management tools"""
        return [
            Tool(
                name="list_models",
                description="List all available models in AI Playground",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Filter by model type",
                            "enum": ["diffusion", "llm", "embedding", "upscaler", "all"],
                            "default": "all"
                        },
                        "status": {
                            "type": "string", 
                            "description": "Filter by model status",
                            "enum": ["installed", "available", "downloading", "all"],
                            "default": "all"
                        },
                        "search": {
                            "type": "string",
                            "description": "Search models by name",
                            "default": ""
                        }
                    }
                }
            ),
            Tool(
                name="model_info",
                description="Get detailed information about a specific model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "ID or name of the model"
                        }
                    },
                    "required": ["model_id"]
                }
            ),
            Tool(
                name="download_model",
                description="Download a model to AI Playground",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "ID or name of the model to download"
                        },
                        "model_url": {
                            "type": "string",
                            "description": "URL to download the model from (optional if model_id is in registry)",
                            "default": ""
                        },
                        "model_type": {
                            "type": "string",
                            "description": "Type of model being downloaded",
                            "enum": ["diffusion", "llm", "embedding", "upscaler"],
                            "default": "diffusion"
                        },
                        "force_redownload": {
                            "type": "boolean",
                            "description": "Force redownload even if model exists",
                            "default": false
                        }
                    },
                    "required": ["model_id"]
                }
            ),
            Tool(
                name="delete_model",
                description="Delete a model from AI Playground",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "ID or name of the model to delete"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirm deletion (required for safety)",
                            "default": false
                        }
                    },
                    "required": ["model_id", "confirm"]
                }
            ),
            Tool(
                name="model_download_status",
                description="Check download status of models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Specific model ID to check (optional)",
                            "default": ""
                        }
                    }
                }
            ),
            Tool(
                name="scan_models",
                description="Scan for new models in the model directories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Type of models to scan for",
                            "enum": ["diffusion", "llm", "embedding", "upscaler", "all"],
                            "default": "all"
                        },
                        "deep_scan": {
                            "type": "boolean",
                            "description": "Perform deep scan including subdirectories",
                            "default": false
                        }
                    }
                }
            )
        ]
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        """Execute model management tool"""
        
        if name == "list_models":
            return await self._list_models(arguments)
        elif name == "model_info":
            return await self._get_model_info(arguments)
        elif name == "download_model":
            return await self._download_model(arguments)
        elif name == "delete_model":
            return await self._delete_model(arguments)
        elif name == "model_download_status":
            return await self._check_download_status(arguments)
        elif name == "scan_models":
            return await self._scan_models(arguments)
        else:
            raise ValueError(f"Unknown model management tool: {name}")
    
    async def _list_models(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """List available models"""
        try:
            params = {
                "type": args.get("model_type", "all"),
                "status": args.get("status", "all")
            }
            if args.get("search"):
                params["search"] = args["search"]
            
            url = urljoin(self.base_url, "/api/models")
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                models = result["data"]
                
                output = "📚 Available Models\n"
                output += "=" * 30 + "\n\n"
                
                if not models:
                    output += "No models found matching the criteria."
                else:
                    # Group models by type
                    model_groups = {}
                    for model in models:
                        model_type = model.get("type", "unknown")
                        if model_type not in model_groups:
                            model_groups[model_type] = []
                        model_groups[model_type].append(model)
                    
                    for model_type, type_models in model_groups.items():
                        output += f"**{model_type.upper()} Models:**\n"
                        for model in type_models:
                            status_icon = "✅" if model.get("status") == "installed" else "📥"
                            output += f"  {status_icon} {model.get('name', 'Unknown')}\n"
                            output += f"    ID: {model.get('id', 'N/A')}\n"
                            output += f"    Status: {model.get('status', 'unknown')}\n"
                            if model.get('size'):
                                output += f"    Size: {model.get('size')}\n"
                            output += "\n"
                
                return [TextContent(type="text", text=output)]
            else:
                return [TextContent(type="text", text=f"❌ Failed to fetch models: {result.get('message', 'Unknown error')}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error listing models: {str(e)}")]