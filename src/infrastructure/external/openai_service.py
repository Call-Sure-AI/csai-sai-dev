# src/infrastructure/external/openai_service.py
"""
OpenAI service implementation for language model operations.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from core.interfaces.external import ILanguageModelService

logger = logging.getLogger(__name__)

class OpenAIService(ILanguageModelService):
    """OpenAI implementation of language model service."""
    
    def __init__(self, api_key: str, default_model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    async def generate_text_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text completion with streaming."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            stream = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating text stream: {e}")
            raise
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        try:
            completion_kwargs = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            if functions:
                completion_kwargs["functions"] = functions
                completion_kwargs["function_call"] = "auto"
            
            response = await self.client.chat.completions.create(**completion_kwargs)
            
            result = {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Add function call info if present
            if hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
                result["function_call"] = {
                    "name": response.choices[0].message.function_call.name,
                    "arguments": response.choices[0].message.function_call.arguments
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            raise
    
    async def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate chat completion with streaming."""
        try:
            completion_kwargs = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
            
            if functions:
                completion_kwargs["functions"] = functions
                completion_kwargs["function_call"] = "auto"
            
            stream = await self.client.chat.completions.create(**completion_kwargs)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "type": "content"
                    }
                elif hasattr(chunk.choices[0].delta, 'function_call') and chunk.choices[0].delta.function_call:
                    yield {
                        "function_call": chunk.choices[0].delta.function_call,
                        "type": "function_call"
                    }
                    
        except Exception as e:
            logger.error(f"Error generating chat stream: {e}")
            raise
    
    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Generate text embeddings."""
        try:
            response = await self.client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self.client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            models = await self.client.models.list()
            return [{"id": model.id, "object": model.object} for model in models.data]
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            model = await self.client.models.retrieve(model_name)
            return {
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {}
    
    async def validate_api_key(self) -> bool:
        """Validate the API key."""
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False