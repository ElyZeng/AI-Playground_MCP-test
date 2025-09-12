"""
LLM Tools for AI Playground MCP Server
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin

import httpx
from mcp.types import Tool, TextContent


class LLMTools:
    """Tools for AI Playground LLM functionality"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)  # 2 minute timeout for LLM
        
    async def get_tools(self) -> List[Tool]:
        """Get all available LLM tools"""
        return [
            Tool(
                name="chat_completion",
                description="Generate chat completion using AI Playground LLM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "Chat messages in OpenAI format",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["system", "user", "assistant"]
                                    },
                                    "content": {
                                        "type": "string"
                                    }
                                },
                                "required": ["role", "content"]
                            }
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM model to use",
                            "default": ""
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 1000,
                            "minimum": 1,
                            "maximum": 4096
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 2.0
                        },
                        "top_p": {
                            "type": "number",
                            "description": "Nucleus sampling parameter",
                            "default": 0.9,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Enable streaming response",
                            "default": false
                        }
                    },
                    "required": ["messages"]
                }
            ),
            Tool(
                name="text_completion",
                description="Generate text completion from prompt",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt for completion"
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM model to use",
                            "default": ""
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 1000
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature",
                            "default": 0.7
                        },
                        "top_p": {
                            "type": "number", 
                            "description": "Nucleus sampling parameter",
                            "default": 0.9
                        },
                        "stop": {
                            "type": "array",
                            "description": "Stop sequences",
                            "items": {"type": "string"},
                            "default": []
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="embedding_generation",
                description="Generate text embeddings using AI Playground",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to generate embeddings for"
                        },
                        "model": {
                            "type": "string",
                            "description": "Embedding model to use",
                            "default": "sentence-transformers/all-MiniLM-L6-v2"
                        },
                        "normalize": {
                            "type": "boolean",
                            "description": "Normalize embeddings to unit length",
                            "default": true
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="rag_search",
                description="Perform RAG (Retrieval Augmented Generation) search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for RAG"
                        },
                        "knowledge_base": {
                            "type": "string",
                            "description": "Knowledge base to search in",
                            "default": "default"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Similarity threshold for results",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "generate_answer": {
                            "type": "boolean",
                            "description": "Generate answer based on retrieved context",
                            "default": true
                        }
                    },
                    "required": ["query"]
                }
                        # 在 get_tools() 方法的 return 列表中添加（第183行前）：
            Tool(
                name="smart_completion",
                description="智能完成 - 集成 MCP 工具調用",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "Chat messages in OpenAI format",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                    "content": {"type": "string"}
                                },
                                "required": ["role", "content"]
                            }
                        },
                        "enable_mcp": {
                            "type": "boolean",
                            "description": "Enable MCP tool integration",
                            "default": true
                        }
                    },
                    "required": ["messages"]
                }
            )
        ]
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        """Execute LLM tool"""
        
        if name == "chat_completion":
            return await self._chat_completion(arguments)
        elif name == "text_completion":
            return await self._text_completion(arguments)
        elif name == "embedding_generation":
            return await self._generate_embeddings(arguments)
        elif name == "rag_search":
            return await self._rag_search(arguments)
        else:
            raise ValueError(f"Unknown LLM tool: {name}")
    
    async def _chat_completion(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Generate chat completion"""
        try:
            payload = {
                "messages": args["messages"],
                "max_tokens": args.get("max_tokens", 1000),
                "temperature": args.get("temperature", 0.7),
                "top_p": args.get("top_p", 0.9),
                "stream": args.get("stream", False)
            }
            
            if args.get("model"):
                payload["model"] = args["model"]
            
            url = urljoin(self.base_url, "/api/llm/chat/completions")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                data = result["data"]
                
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    
                    # Add usage statistics if available
                    usage_info = ""
                    if "usage" in data:
                        usage = data["usage"]
                        usage_info = f"\n\nTokens used: {usage.get('total_tokens', 'unknown')} (prompt: {usage.get('prompt_tokens', 'unknown')}, completion: {usage.get('completion_tokens', 'unknown')})"
                    
                    return [TextContent(
                        type="text",
                        text=content + usage_info
                    )]
                else:
                    return [TextContent(type="text", text="No completion generated")]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Chat completion failed: {error_msg}")]
                
        except httpx.RequestError as e:
            return [TextContent(type="text", text=f"Network error: {str(e)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error in chat completion: {str(e)}")]
    
    async def _text_completion(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Generate text completion"""
        try:
            payload = {
                "prompt": args["prompt"],
                "max_tokens": args.get("max_tokens", 1000),
                "temperature": args.get("temperature", 0.7),
                "top_p": args.get("top_p", 0.9),
                "stop": args.get("stop", [])
            }
            
            if args.get("model"):
                payload["model"] = args["model"]
            
            url = urljoin(self.base_url, "/api/llm/completions")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                data = result["data"]
                
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    text = choice.get("text", "")
                    
                    # Add usage statistics if available
                    usage_info = ""
                    if "usage" in data:
                        usage = data["usage"]
                        usage_info = f"\n\nTokens used: {usage.get('total_tokens', 'unknown')}"
                    
                    return [TextContent(
                        type="text",
                        text=f"Completion: {text}{usage_info}"
                    )]
                else:
                    return [TextContent(type="text", text="No completion generated")]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Text completion failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error in text completion: {str(e)}")]
    
    async def _generate_embeddings(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Generate text embeddings"""
        try:
            payload = {
                "input": args["text"],
                "model": args.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                "normalize": args.get("normalize", True)
            }
            
            url = urljoin(self.base_url, "/api/llm/embeddings")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                data = result["data"]
                
                if "data" in data and len(data["data"]) > 0:
                    embedding_data = data["data"][0]
                    embedding = embedding_data.get("embedding", [])
                    
                    embedding_info = f"Generated embedding for text: '{args['text'][:50]}...'\n"
                    embedding_info += f"Model: {args.get('model', 'default')}\n"
                    embedding_info += f"Dimensions: {len(embedding)}\n"
                    embedding_info += f"Sample values: {embedding[:5]}..."
                    
                    return [TextContent(type="text", text=embedding_info)]
                else:
                    return [TextContent(type="text", text="No embeddings generated")]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"Embedding generation failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating embeddings: {str(e)}")]
    
    async def _rag_search(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Perform RAG search"""
        try:
            payload = {
                "query": args["query"],
                "knowledge_base": args.get("knowledge_base", "default"),
                "top_k": args.get("top_k", 5),
                "threshold": args.get("threshold", 0.7),
                "generate_answer": args.get("generate_answer", True)
            }
            
            url = urljoin(self.base_url, "/api/llm/rag/search")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and result.get("data"):
                data = result["data"]
                
                output = f"RAG Search Results for: '{args['query']}'\n"
                output += f"Knowledge Base: {args.get('knowledge_base', 'default')}\n\n"
                
                # Display retrieved documents
                if "documents" in data:
                    output += "Retrieved Documents:\n"
                    for i, doc in enumerate(data["documents"], 1):
                        output += f"{i}. Score: {doc.get('score', 'N/A'):.3f}\n"
                        output += f"   Content: {doc.get('content', '')[:200]}...\n\n"
                
                # Display generated answer if available
                if args.get("generate_answer", True) and "answer" in data:
                    output += f"Generated Answer:\n{data['answer']}\n"
                
                return [TextContent(type="text", text=output)]
            else:
                error_msg = result.get("message", "Unknown error")
                return [TextContent(type="text", text=f"RAG search failed: {error_msg}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Error in RAG search: {str(e)}")]
    
    def __del__(self):
        """Cleanup HTTP client"""
        if hasattr(self, 'client'):
            asyncio.create_task(self.client.aclose())
        # 在 mcp/tools/llm_tools.py 中添加
    async def _mcp_tool_call(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """調用 MCP 工具作為 LLM 的擴展功能"""
        try:
            tool_name = args.get("tool_name")
            tool_args = args.get("tool_args", {})
            
            # 根據工具名稱調用相應的 MCP 工具
            if tool_name == "generate_image":
                from .image_generation import ImageGenerationTools
                image_tools = ImageGenerationTools(self.base_url)
                return await image_tools.call_tool("text_to_image", tool_args)
            elif tool_name == "manage_model":
                from .model_management import ModelManagementTools  
                model_tools = ModelManagementTools(self.base_url)
                return await model_tools.call_tool("list_models", tool_args)
            # 添加更多 MCP 工具...
            
        except Exception as e:
            return [TextContent(type="text", text=f"MCP 工具調用錯誤: {str(e)}")]
            
    async def _smart_completion(self, args: Dict[str, Any]) -> Sequence[TextContent]:
    """智能完成 - 集成 MCP 工具調用"""
    # 1. 先進行 LLM 推理
    llm_result = await self._chat_completion(args)
    
    if not args.get("enable_mcp", True):
        return llm_result
    
    # 2. 分析響應，檢測工具調用需求
    response_text = llm_result[0].text
    
    # 3. 如果需要，調用相應的 MCP 工具
    additional_results = []
    
    if self._needs_image_generation(response_text):
        image_result = await self._generate_image_from_response(response_text)
        additional_results.extend(image_result)
    
    if self._needs_model_management(response_text):
        model_result = await self._handle_model_request(response_text)
        additional_results.extend(model_result)
    
    # 合併結果
    return llm_result + additional_results

def _needs_image_generation(self, text: str) -> bool:
    """檢測是否需要圖像生成"""
    keywords = ["生成圖片", "畫一張", "創建圖像", "generate image", "create picture"]
    return any(keyword in text.lower() for keyword in keywords)

def _needs_model_management(self, text: str) -> bool:
    """檢測是否需要模型管理"""
    keywords = ["列出模型", "下載模型", "管理模型", "list models", "download model"]
    return any(keyword in text.lower() for keyword in keywords)

async def _generate_image_from_response(self, text: str) -> Sequence[TextContent]:
    """從響應中提取圖像生成請求並執行"""
    try:
        # 簡單的提示詞提取邏輯
        prompt = self._extract_image_prompt(text)
        
        from .image_generation import ImageGenerationTools
        image_tools = ImageGenerationTools(self.base_url)
        return await image_tools.call_tool("text_to_image", {"prompt": prompt})
    except Exception as e:
        return [TextContent(type="text", text=f"圖像生成失敗: {str(e)}")]

async def _handle_model_request(self, text: str) -> Sequence[TextContent]:
    """處理模型管理請求"""
    try:
        from .model_management import ModelManagementTools
        model_tools = ModelManagementTools(self.base_url)
        return await model_tools.call_tool("list_models", {})
    except Exception as e:
        return [TextContent(type="text", text=f"模型管理失敗: {str(e)}")]

def _extract_image_prompt(self, text: str) -> str:
    """從文本中提取圖像生成提示詞"""
    # 簡單的提取邏輯，可以根據需要改進
    import re
    patterns = [
        r"生成圖片[：:]\s*([^。\n]+)",
        r"畫一張\s*([^。\n]+)",
        r"create\s+(?:an?\s+)?image\s+of\s+([^.\n]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 如果沒有匹配到特定格式，返回一個默認提示
    return "a beautiful landscape"