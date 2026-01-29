# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Redis utilities for Prism multi-model serving.

Provides synchronous and asynchronous Redis clients for:
- Request queue management
- Inter-process communication
- Model status updates
"""

import logging
import pickle
from typing import Any, List, Optional

import redis

logger = logging.getLogger(__name__)


class AsyncRedisClient:
    """Asynchronous Redis client for non-blocking operations."""

    def __init__(self, host: str, port: int, db: int):
        self.host = host
        self.port = port
        self.db = db
        self.client = redis.asyncio.Redis(host=host, port=port, db=db)

    async def reconnect(self):
        """Reconnect to Redis server if connection is lost."""
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass  # Ignore errors during close
        self.client = redis.asyncio.Redis(
            host=self.host, port=self.port, db=self.db
        )
        # Test connection
        await self.client.ping()

    async def send_pyobj(self, key: str, obj: Any):
        """Send a Python object to a Redis queue."""
        obj_bytes = pickle.dumps(obj)
        await self.client.rpush(key, obj_bytes)

    async def recv_pyobj_non_block(self, key: str, count: int = 1) -> List[Any]:
        """Receive Python objects from a Redis queue (non-blocking)."""
        obj_bytes = await self.client.lpop(key, count=count)
        if isinstance(obj_bytes, list):
            return [pickle.loads(obj) for obj in obj_bytes]
        if obj_bytes is None:
            return []
        return [pickle.loads(obj_bytes)]

    async def recv_pyobj_block(self, key: str) -> Any:
        """Receive a Python object from a Redis queue (blocking)."""
        _, obj_bytes = await self.client.blpop(key)
        return pickle.loads(obj_bytes)

    async def close(self):
        """Close the Redis connection."""
        await self.client.close()


class RedisClient:
    """Synchronous Redis client for blocking operations.
    
    Matches prism-old implementation exactly.
    """

    def __init__(self, host: str, port: int, db: int):
        self.host = host
        self.port = port
        self.db = db
        self.client = redis.Redis(host=host, port=port, db=db)

    def reconnect(self):
        """Reconnect to Redis server if connection is lost."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
        self.client.ping()

    def clear_queue(self):
        """Clear all queues in the database."""
        self.client.flushdb()

    def send_pyobj(self, key: str, obj: Any):
        """Send a Python object to a Redis queue."""
        obj_bytes = pickle.dumps(obj)
        self.client.rpush(key, obj_bytes)

    def recv_pyobj_non_block(self, key: str, count: int = 1) -> List[Any]:
        """Receive Python objects from a Redis queue (non-blocking)."""
        obj_bytes = self.client.lpop(key, count=count)
        if isinstance(obj_bytes, list):
            return [pickle.loads(obj) for obj in obj_bytes]
        if obj_bytes is None:
            return []
        return [pickle.loads(obj_bytes)]

    def get_queue_length(self, key: str) -> int:
        """Get the length of a Redis queue."""
        return self.client.llen(key)

    def recv_pyobj_block(self, key: str) -> Any:
        """Receive a Python object from a Redis queue (blocking)."""
        _, obj_bytes = self.client.blpop(key)
        return pickle.loads(obj_bytes)

    def pop_all(self, key: str) -> List[Any]:
        """Pop all items from a Redis queue atomically."""
        length = self.client.llen(key)
        if length == 0:
            return []

        pipe = self.client.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results = pipe.execute()
        return [pickle.loads(obj) for obj in results[0]]

    def close(self):
        """Close the Redis connection."""
        self.client.close()


__all__ = ["RedisClient", "AsyncRedisClient"]
