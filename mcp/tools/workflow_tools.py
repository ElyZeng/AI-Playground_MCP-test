"""
Workflow Tools for AI Playground MCP Server
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin

import httpx
from mcp.types import Tool, TextContent


class WorkflowTools:
    """Tools for AI Playground ComfyUI workflow functionality"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for workflows
        
    async def get_tools(self) -> List[Tool]:
        """Get all available workflow tools"""
        return [
            Tool(
                name="list_workflows",
                description="List all available ComfyUI workflows in AI Playground",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Filter workflows by category",
                            "default": ""
                        },
                        "search": {
                            "type": "string",
                            "description": "Search workflows by name or description",
                            "default": ""
                        }
                    }
                }
            ),
            Tool(
                name="workflow_execute",
                description="Execute a ComfyUI workflow with parameters",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "ID of the workflow to execute"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the workflow",
                            "default": {}
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Desired output format",
                            "enum": ["image", "video", "audio", "text"],
                            "default": "image"
                        }
                    },
                    "required": ["workflow_id"]
                }
            ),
            Tool(
                name="workflow_validate",
                description="Validate workflow dependencies and requirements",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "ID of the workflow to validate"
                        },
                        "check_models": {
                            "type": "boolean",
                            "description": "Check if required models are available",
                            "default": true
                        },
                        "check_nodes": {
                            "type": "boolean",
                            "description": "Check if required custom nodes are installed",
                            "default": true
                        }
                    },
                    "required": ["workflow_id"]
                }
            ),
            Tool(
                name="workflow_get_info",
                description="Get detailed information about a specific workflow",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "ID of the workflow to get info for"
                        }
                    },
                    "required": ["workflow_id"]
                }
            ),
            Tool(
                name="workflow_install_requirements",
                description="Install missing requirements for a workflow",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "ID of the workflow"
                        },
                        "install_models": {
                            "type": "boolean",
                            "description": "Install missing models",
                            "default": true
                        },
                        "install_nodes": {
                            "type": "boolean",
                            "description": "Install missing custom nodes",
                            "default": true
                        }
                    },
                    "required": ["workflow_id"]
                }
            )
        ]
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        """Execute workflow tool"""
        
        if name == "list_workflows":
            return await self._list_workflows(arguments)
        elif name == "workflow_execute":
            return await self._execute_workflow(arguments)
        elif name == "workflow_validate":
            return await self._validate_workflow(arguments)
        elif name == "workflow_get_info":
            return await self._get_workflow_info(arguments)
        elif name == "workflow_install_requirements":
            return await self._install_requirements(arguments)
        else:
            raise ValueError(f"Unknown workflow tool: {name}")
    
    async def _list_workflows(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """List available workflows"""
        try:
            params = {}
            if args.get("category"):
                params["category"] = args["category"]
            if args.get("search"):
                params["search"] = args["search"]
            
            url = urljoin(self.base_url, "/api/comfyui/workflows")
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                workflows = result["data"]
                
                output = "Available ComfyUI Workflows:\n\n"
                
                if not workflows:
                    output += "No workflows found."
                else:
                    for workflow in workflows:
                        output += f"🔧 **{workflow.get('name', workflow.get('id', 'Unknown'))}**\n"
                        output += f"   ID: {workflow.get('id', 'N/A')}\n"
                        if workflow.get('description'):
                            output += f"   Description: {workflow['description']}\n"
                        if workflow.get('category'):
                            output += f"   Category: {workflow['category']}\n"
                        if workflow.get('tags'):
                            output += f"   Tags: {', '.join(workflow['tags'])}\n"
                        output += "\n"
                
                return [TextContent(type="text", text=output)]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Failed to list workflows: {error_msg}")]
                
        except httpx.RequestError as e:
            return [TextContent(type="text", text=f"Network error: {str(e)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing workflows: {str(e)}")]
    
    async def _execute_workflow(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Execute workflow"""
        try:
            payload = {
                "workflow_id": args["workflow_id"],
                "parameters": args.get("parameters", {}),
                "output_format": args.get("output_format", "image")
            }
            
            url = urljoin(self.base_url, "/api/comfyui/execute")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                data = result["data"]
                
                output = f"✅ Workflow '{args['workflow_id']}' executed successfully!\n\n"
                
                if "execution_id" in data:
                    output += f"Execution ID: {data['execution_id']}\n"
                
                if "outputs" in data:
                    output += "Outputs:\n"
                    for i, output_data in enumerate(data["outputs"], 1):
                        output += f"  {i}. Type: {output_data.get('type', 'unknown')}\n"
                        if output_data.get('path'):
                            output += f"     Path: {output_data['path']}\n"
                        if output_data.get('url'):
                            output += f"     URL: {output_data['url']}\n"
                
                if "execution_time" in data:
                    output += f"\nExecution time: {data['execution_time']:.2f} seconds"
                
                return [TextContent(type="text", text=output)]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Workflow execution failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error executing workflow: {str(e)}")]
    
    async def _validate_workflow(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Validate workflow dependencies"""
        try:
            payload = {
                "workflow_id": args["workflow_id"],
                "check_models": args.get("check_models", True),
                "check_nodes": args.get("check_nodes", True)
            }
            
            url = urljoin(self.base_url, "/api/comfyui/validate-workflow")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                data = result["data"]
                
                output = f"Workflow Validation: {args['workflow_id']}\n\n"
                
                is_valid = data.get("valid", False)
                output += f"Status: {'✅ Valid' if is_valid else '❌ Invalid'}\n\n"
                
                if "missingModels" in data and data["missingModels"]:
                    output += "Missing Models:\n"
                    for model in data["missingModels"]:
                        output += f"  - {model.get('name', 'Unknown')}\n"
                        if model.get('url'):
                            output += f"    Download URL: {model['url']}\n"
                    output += "\n"
                
                if "missingNodes" in data and data["missingNodes"]:
                    output += "Missing Custom Nodes:\n"
                    for node in data["missingNodes"]:
                        output += f"  - {node.get('name', 'Unknown')}\n"
                        if node.get('repository'):
                            output += f"    Repository: {node['repository']}\n"
                    output += "\n"
                
                if is_valid:
                    output += "All requirements are satisfied! ✅"
                else:
                    output += "Please install missing dependencies to use this workflow."
                
                return [TextContent(type="text", text=output)]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Workflow validation failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error validating workflow: {str(e)}")]
    
    async def _get_workflow_info(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Get detailed workflow information"""
        try:
            workflow_id = args["workflow_id"]
            
            url = urljoin(self.base_url, f"/api/comfyui/workflows/{workflow_id}")
            response = await self.client.get(url)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                workflow = result["data"]
                
                output = f"📋 Workflow Information: {workflow.get('name', workflow_id)}\n"
                output += "=" * 50 + "\n\n"
                
                if workflow.get('description'):
                    output += f"Description: {workflow['description']}\n\n"
                
                if workflow.get('author'):
                    output += f"Author: {workflow['author']}\n"
                if workflow.get('version'):
                    output += f"Version: {workflow['version']}\n"
                if workflow.get('category'):
                    output += f"Category: {workflow['category']}\n"
                if workflow.get('tags'):
                    output += f"Tags: {', '.join(workflow['tags'])}\n"
                
                output += "\n"
                
                if workflow.get('parameters'):
                    output += "Parameters:\n"
                    for param in workflow['parameters']:
                        output += f"  • {param.get('name', 'Unknown')}: {param.get('type', 'any')}\n"
                        if param.get('description'):
                            output += f"    Description: {param['description']}\n"
                        if param.get('default') is not None:
                            output += f"    Default: {param['default']}\n"
                    output += "\n"
                
                if workflow.get('requiredModels'):
                    output += "Required Models:\n"
                    for model in workflow['requiredModels']:
                        output += f"  • {model.get('name', 'Unknown')}\n"
                        if model.get('type'):
                            output += f"    Type: {model['type']}\n"
                    output += "\n"
                
                if workflow.get('customNodes'):
                    output += "Required Custom Nodes:\n"
                    for node in workflow['customNodes']:
                        output += f"  • {node}\n"
                    output += "\n"
                
                return [TextContent(type="text", text=output)]
            else:
                error_msg = result.get("message", "Workflow not found")
                return [TextContent(type="text", text=f"Failed to get workflow info: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting workflow info: {str(e)}")]
    
    async def _install_requirements(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Install missing workflow requirements"""
        try:
            payload = {
                "workflow_id": args["workflow_id"],
                "install_models": args.get("install_models", True),
                "install_nodes": args.get("install_nodes", True)
            }
            
            url = urljoin(self.base_url, "/api/comfyui/install-requirements")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0:
                output = f"🔧 Installing requirements for workflow: {args['workflow_id']}\n\n"
                
                if result.get("data"):
                    data = result["data"]
                    
                    if "installedModels" in data:
                        models = data["installedModels"]
                        if models:
                            output += "Installed Models:\n"
                            for model in models:
                                output += f"  ✅ {model}\n"
                            output += "\n"
                    
                    if "installedNodes" in data:
                        nodes = data["installedNodes"]
                        if nodes:
                            output += "Installed Custom Nodes:\n"
                            for node in nodes:
                                output += f"  ✅ {node}\n"
                            output += "\n"
                    
                    if "skipped" in data:
                        skipped = data["skipped"]
                        if skipped:
                            output += "Skipped (already installed):\n"
                            for item in skipped:
                                output += f"  ⏭️ {item}\n"
                            output += "\n"
                    
                    if "errors" in data:
                        errors = data["errors"]
                        if errors:
                            output += "Errors:\n"
                            for error in errors:
                                output += f"  ❌ {error}\n"
                            output += "\n"
                
                output += "Installation completed! You may need to restart ComfyUI backend for changes to take effect."
                
                return [TextContent(type="text", text=output)]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Requirements installation failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error installing requirements: {str(e)}")]
    
    def __del__(self):
        """Cleanup HTTP client"""
        if hasattr(self, 'client'):
            asyncio.create_task(self.client.aclose())