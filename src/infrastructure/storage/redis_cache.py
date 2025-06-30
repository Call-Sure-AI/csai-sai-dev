# src/infrastructure/storage/redis_cache.py
"""
Redis cache service implementation.
"""

import logging
import json
import pickle
from typing import Optional, List, Dict, Any
import aioredis
from aioredis import Redis

from core.interfaces.external import ICacheService

logger = logging.getLogger(__name__)

class RedisCacheService(ICacheService):
    """Redis implementation of cache service."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        decode_responses: bool = True
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.decode_responses = decode_responses
        self._redis: Optional[Redis] = None
    
    async def _get_redis(self) -> Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                decode_responses=self.decode_responses
            )
        return self._redis
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        else:
            # Use pickle for complex objects
            return pickle.dumps(value).hex()
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                # Try pickle
                return pickle.loads(bytes.fromhex(value))
            except (ValueError, pickle.PickleError):
                return value
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            redis = await self._get_redis()
            value = await redis.get(key)
            if value is None:
                return None
            return self._deserialize(value)
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value with optional TTL."""
        try:
            redis = await self._get_redis()
            serialized_value = self._serialize(value)
            
            if ttl:
                await redis.setex(key, ttl, serialized_value)
            else:
                await redis.set(key, serialized_value)
            
            return True
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            redis = await self._get_redis()
            result = await redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            redis = await self._get_redis()
            result = await redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error checking existence of key {key} in Redis: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values by keys."""
        try:
            redis = await self._get_redis()
            values = await redis.mget(keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
            
            return result
        except Exception as e:
            logger.error(f"Error getting multiple keys from Redis: {e}")
            return {}
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple key-value pairs."""
        try:
            redis = await self._get_redis()
            
            # Serialize all values
            serialized_mapping = {
                key: self._serialize(value)
                for key, value in mapping.items()
            }
            
            if ttl:
                # Use pipeline for TTL
                pipe = redis.pipeline()
                for key, value in serialized_mapping.items():
                    pipe.setex(key, ttl, value)
                await pipe.execute()
            else:
                await redis.mset(serialized_mapping)
            
            return True
        except Exception as e:
            logger.error(f"Error setting multiple keys in Redis: {e}")
            return False
    
    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys and return count deleted."""
        try:
            redis = await self._get_redis()
            result = await redis.delete(*keys)
            return result
        except Exception as e:
            logger.error(f"Error deleting multiple keys from Redis: {e}")
            return 0
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        try:
            redis = await self._get_redis()
            result = await redis.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Error incrementing key {key} in Redis: {e}")
            raise
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key."""
        try:
            redis = await self._get_redis()
            result = await redis.expire(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Error setting expiration for key {key} in Redis: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(pattern)
            if keys:
                result = await redis.delete(*keys)
                return result
            return 0
        except Exception as e:
            logger.error(f"Error clearing pattern {pattern} in Redis: {e}")
            return 0