#!/usr/bin/env python3
"""
AI Playground MCP Server
Provides Model Context Protocol server functionality for AI Playground
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    GetPromptRequest,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from tools.image_generation import ImageGenerationTools
from tools.llm_tools import LLMTools
from tools.workflow_tools import WorkflowTools
from tools.model_management import ModelManagementTools
from tools.service_management import ServiceManagementTools


class AIPlaygroundMCPServer:
    """MCP Server for AI Playground functionality"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.server = Server("ai-playground")
        self.config = self._load_config(config_path)
        self.base_url = self.config.get("base_url", "http://127.0.0.1:8188")
        self.services = {}
        
        # Initialize tool modules
        self.image_tools = ImageGenerationTools(self.base_url)
        self.llm_tools = LLMTools(self.base_url)
        self.workflow_tools = WorkflowTools(self.base_url)
        self.model_tools = ModelManagementTools(self.base_url)
        self.service_tools = ServiceManagementTools(self.base_url)
        
        self._register_handlers()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load MCP server configuration"""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "mcp_config.json"
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _register_handlers(self):
        """Register all MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available MCP tools"""
            tools = []
            
            # Image generation tools
            tools.extend(await self.image_tools.get_tools())
            
            # LLM tools
            tools.extend(await self.llm_tools.get_tools())
            
            # Workflow tools
            tools.extend(await self.workflow_tools.get_tools())
            
            # Model management tools
            tools.extend(await self.model_tools.get_tools())
            
            # Service management tools
            tools.extend(await self.service_tools.get_tools())
            
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool execution requests"""
            if arguments is None:
                arguments = {}
                
            self.logger.info(f"Calling tool: {name} with arguments: {arguments}")
            
            try:
                # Route to appropriate tool handler
                if name.startswith("generate_") or name.startswith("image_"):
                    return await self.image_tools.call_tool(name, arguments)
                elif name.startswith("chat_") or name.startswith("text_") or name.startswith("embedding_"):
                    return await self.llm_tools.call_tool(name, arguments)
                elif name.startswith("workflow_") or name.startswith("list_workflows"):
                    return await self.workflow_tools.call_tool(name, arguments)
                elif name.startswith("model_") or name.startswith("download_") or name.startswith("list_models"):
                    return await self.model_tools.call_tool(name, arguments)
                elif name.startswith("service_") or name.startswith("list_services"):
                    return await self.service_tools.call_tool(name, arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                self.logger.error(f"Error executing tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def check_service_health(self) -> Dict[str, bool]:
        """Check health of AI Playground services"""
        health_status = {}
        
        # Check main AI backend
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/health", timeout=5.0)
                health_status["ai_backend"] = response.status_code == 200
        except Exception:
            health_status["ai_backend"] = False
            
        # Check other services if configured
        for service_name, service_config in self.config.get("services", {}).items():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{service_config['url']}/health", timeout=5.0)
                    health_status[service_name] = response.status_code == 200
            except Exception:
                health_status[service_name] = False
                
        return health_status

    async def run(self):
        """Run the MCP server"""
        self.logger.info("Starting AI Playground MCP Server...")
        
        # Check service health before starting
        health = await self.check_service_health()
        self.logger.info(f"Service health check: {health}")
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, 
                write_stream, 
                InitializationOptions(
                    server_name="ai-playground",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Playground MCP Server")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = AIPlaygroundMCPServer(args.config)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
    except Exception as e:
        print(f"Error running MCP server: {e}")
        sys.exit(1)