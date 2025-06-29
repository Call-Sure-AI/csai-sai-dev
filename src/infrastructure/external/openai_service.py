# src/infrastructure/external/openai_service.py
import asyncio
import logging
from typing import List, Dict, AsyncGenerator, Optional
from openai import AsyncOpenAI

from core.interfaces.external import IAIService

logger = logging.getLogger(__name__)

class OpenAIService(IAIService):
    """OpenAI service implementation for AI/LLM operations"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = "gpt-4"
        self.default_embedding_model = "text-embedding-3-small"
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        agent_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming AI response"""
        try:
            # Add system prompt if provided
            formatted_messages = []
            if agent_prompt:
                formatted_messages.append({"role": "system", "content": agent_prompt})
            formatted_messages.extend(messages)
            
            # Create streaming completion
            stream = await self.client.chat.completions.create(
                model=self.default_model,
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            
            # Yield tokens as they arrive
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            yield f"Error: {str(e)}"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.default_embedding_model
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []