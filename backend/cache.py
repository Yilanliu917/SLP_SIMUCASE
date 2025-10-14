"""
Redis cache configuration and utilities
"""
import redis.asyncio as redis
from typing import Optional, Any
import json
import os
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Default 1 hour

# Redis client singleton
_redis_client: Optional[redis.Redis] = None


async def get_cache() -> redis.Redis:
    """Get Redis client instance"""
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=10
        )

    return _redis_client


async def close_cache():
    """Close Redis connection"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


def cache_response(ttl: int = CACHE_TTL, key_prefix: str = ""):
    """
    Decorator to cache function responses in Redis
    Usage:
        @cache_response(ttl=600, key_prefix="analytics")
        async def get_analytics():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"

            try:
                cache = await get_cache()

                # Try to get from cache
                cached_value = await cache.get(cache_key)
                if cached_value:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return json.loads(cached_value)

                # Call function
                result = await func(*args, **kwargs)

                # Store in cache
                await cache.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                logger.debug(f"Cached result for key: {cache_key}")

                return result

            except Exception as e:
                logger.warning(f"Cache operation failed: {e}")
                # If cache fails, still return the function result
                return await func(*args, **kwargs)

        return wrapper
    return decorator


async def invalidate_cache(pattern: str):
    """Invalidate cache keys matching pattern"""
    try:
        cache = await get_cache()
        keys = await cache.keys(pattern)
        if keys:
            await cache.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache keys matching: {pattern}")
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
