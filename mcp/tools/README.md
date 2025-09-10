# AI Playground MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for AI Playground, enabling external AI assistants to interact with all AI Playground functionalities through a standardized protocol.

## 🚀 Features

### Image Generation
- **Text-to-Image**: Generate images from text prompts
- **Image-to-Image**: Transform existing images based on prompts  
- **AI Upscaling**: Enhance image resolution using AI models
- **Image Enhancement**: Improve image quality and details

### LLM Capabilities
- **Chat Completion**: Interactive conversation with LLM models
- **Text Completion**: Complete text based on prompts
- **Embeddings**: Generate text embeddings for similarity search
- **RAG Search**: Retrieval Augmented Generation with knowledge bases

### Workflow Management  
- **ComfyUI Integration**: List, execute, and validate ComfyUI workflows
- **Dependency Management**: Automatic installation of required models and nodes
- **Workflow Information**: Detailed workflow specifications and requirements

### Model Management
- **Model Discovery**: List available models across different categories
- **Model Downloads**: Download and install new models
- **Model Information**: Get detailed model specifications
- **Storage Management**: Scan and manage model files

### Service Management
- **Health Monitoring**: Check status of all backend services
- **Service Control**: Start/stop backend services
- **Multi-Backend Support**: Coordinate AI Backend, ComfyUI, LlamaCPP, OpenVINO, Ollama

## 🛠️ Installation

1. **Install Dependencies**:
```bash
cd mcp
pip install -r requirements.txt
```

2. **Configure Services** (optional):
Edit `config/mcp_config.json` to match your backend service URLs and settings.

3. **Start MCP Server**:
```bash
python server.py
```

For debug mode:
```bash
python server.py --debug
```

## 🔧 Configuration

### MCP Server Configuration (`config/mcp_config.json`)

```json
{
  "base_url": "http://127.0.0.1:8188",
  "services": {
    "ai_backend": {
      "url": "http://127.0.0.1:8188",
      "type": "flask"
    },
    "comfyui": {
      "url": "http://127.0.0.1:8189",
      "type": "comfyui"  
    }
  }
}
```

### Service URLs
The MCP server automatically discovers and integrates with:
- **AI Backend**: Main Flask-based inference service (port 8188)
- **ComfyUI**: Node-based workflow engine (port 8189)
- **LlamaCPP**: CPU-optimized LLM service (port 8190)  
- **OpenVINO**: Intel-optimized inference (port 8191)
- **Ollama**: Local LLM management (port 11434)

## 🎯 MCP Tools Available

### Image Generation Tools
- `generate_image` - Create images from text prompts
- `image_to_image` - Transform images with prompts
- `upscale_image` - AI-powered image upscaling
- `enhance_image` - Improve image quality

### LLM Tools  
- `chat_completion` - Interactive chat with AI
- `text_completion` - Complete text prompts
- `embedding_generation` - Generate text embeddings
- `rag_search` - Search knowledge bases

### Workflow Tools
- `list_workflows` - List available ComfyUI workflows
- `workflow_execute` - Run specific workflows
- `workflow_validate` - Check workflow dependencies
- `workflow_get_info` - Get workflow details
- `workflow_install_requirements` - Install missing dependencies

### Model Management Tools
- `list_models` - List available AI models
- `model_info` - Get model specifications
- `download_model` - Download new models
- `delete_model` - Remove models
- `model_download_status` - Check download progress
- `scan_models` - Discover new models

### Service Management Tools
- `list_services` - List all backend services
- `service_status` - Check service health
- `start_service` - Start backend services
- `stop_service` - Stop backend services
- `restart_service` - Restart services

## 🔌 Integration with External AI Assistants

The MCP server enables external AI assistants (like Claude, GPT-4, etc.) to:

1. **Generate Images**: Create custom artwork and visuals
2. **Process Text**: Use local LLMs for private conversations  
3. **Manage Workflows**: Execute complex AI pipelines
4. **Handle Models**: Download and manage AI models
5. **Control Services**: Monitor and manage backend services

### Example MCP Tool Usage

```python
# Generate an image
await mcp_client.call_tool("generate_image", {
    "prompt": "A beautiful sunset over mountains",
    "width": 1024,
    "height": 768,
    "steps": 30
})

# Chat with local LLM
await mcp_client.call_tool("chat_completion", {
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 500
})

# Execute ComfyUI workflow  
await mcp_client.call_tool("workflow_execute", {
    "workflow_id": "text2img_basic",
    "parameters": {
        "prompt": "A robot in a futuristic city"
    }
})
```

## 🏗️ Architecture

```
AI Playground MCP Server
├── server.py                 # Main MCP server
├── tools/                    # MCP tool implementations
│   ├── image_generation.py   # Image generation tools
│   ├── llm_tools.py