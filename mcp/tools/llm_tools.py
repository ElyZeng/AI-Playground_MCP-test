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