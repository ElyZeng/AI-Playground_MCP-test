
class LLMMessageHandler:
    """處理 LLM 消息中的 MCP 工具調用"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.mcp_tools = {
            "image": ImageGenerationTools(base_url),
            "model": ModelManagementTools(base_url),
            # 更多工具...
        }
    
    async def process_message(self, messages: List[Dict], llm_response: str):
        """分析 LLM 響應，檢測是否需要調用 MCP 工具"""
        # 檢測特定關鍵詞或格式
        if "生成圖片:" in llm_response:
            prompt = self.extract_image_prompt(llm_response)
            return await self.mcp_tools["image"].call_tool("text_to_image", {"prompt": prompt})
        
        return llm_response