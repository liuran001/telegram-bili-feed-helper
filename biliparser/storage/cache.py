import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import redis.asyncio as redis

from ..utils import logger

LOCAL_FILE_PATH = Path(os.environ.get("LOCAL_TEMP_FILE_PATH", str(Path.cwd())))
UPLOAD_LOCK_PREFIX = "upload_lock:"


class FakeLock:
    def __init__(self, store, lock_key, timeout=10):
        self.store = store
        self.lock_key = lock_key
        self.timeout = timeout
        self._acquired = False

    async def acquire(self):
        current_time = int(time.time())
        lock_value = await self.store.get(self.lock_key)
        if lock_value:
            if current_time - float(lock_value) > self.timeout:
                await self.store.set(self.lock_key, str(current_time))
                self._acquired = True
                return True
            return False
        await self.store.set(self.lock_key, str(current_time))
        self._acquired = True
        return True

    async def release(self):
        if self._acquired:
            await self.store.delete(self.lock_key)
            self._acquired = False

    async def __aenter__(self):
        while not await self.acquire():
            await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class FakeRedis:
    def __init__(self):
        self.cache_file = LOCAL_FILE_PATH / "cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[Any, Any]:
        try:
            with self.cache_file.open(encoding="utf-8") as f:
                result = json.load(f)
                if isinstance(result, dict) and result.get("__version") == 2:
                    return result
        except (OSError, json.JSONDecodeError):
            pass
        return {"__version": 2}

    def _save_cache(self):
        with self.cache_file.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)

    async def get(self, key: str):
        if key == "__version":
            return None
        target = self.cache.get(key)
        if target and isinstance(target, dict):
            if target.get("timeout") and target["timeout"] < int(time.time()):
                del self.cache[key]
                self._save_cache()
                return None
            return target.get("value")
        return None

    async def set(
        self, key: str, value: str | bytes, ex: int | None = None, nx: bool | None = None, *args, **kwargs
    ) -> None:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if nx and key in self.cache:
            return
        self.cache[key] = {"value": value}
        if isinstance(ex, int):
            self.cache[key]["timeout"] = int(time.time()) + ex
        self._save_cache()

    async def incr(self, key: str) -> int:
        value = await self.get(key)
        count = int(value or 0) + 1
        timeout = self.cache.get(key, {}).get("timeout")
        self.cache[key] = {"value": str(count)}
        if timeout:
            self.cache[key]["timeout"] = timeout
        self._save_cache()
        return count

    async def expire(self, key: str, time_seconds: int) -> bool:
        if key not in self.cache:
            return False
        self.cache[key]["timeout"] = int(time.time()) + time_seconds
        self._save_cache()
        return True

    async def ttl(self, key: str) -> int:
        if await self.get(key) is None:
            return -2
        timeout = self.cache.get(key, {}).get("timeout")
        if not timeout:
            return -1
        return max(0, timeout - int(time.time()))

    async def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
            self._save_cache()

    async def scan_iter(self, match: str):
        prefix = match.rstrip("*")
        for key in list(self.cache):
            if key.startswith(prefix):
                yield key

    def lock(self, key: str, timeout: int = 3600):
        return FakeLock(self, key, timeout)


class RedisCache:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            if os.environ.get("REDIS_URL"):
                cls.instance = redis.Redis.from_url(os.environ["REDIS_URL"])
            else:
                cls.instance = FakeRedis()
        return cls.instance


def upload_lock_key(url: str) -> str:
    """媒体处理锁的 key。带前缀才能与同库的缓存 key 区分，支持启动时安全批量清理。"""
    return f"{UPLOAD_LOCK_PREFIX}{url}"


async def clear_stale_upload_locks(store=None) -> int:
    """清理上次进程异常退出遗留的媒体处理锁。

    锁 TTL 长达 ``2 * CACHES_TIMER["LOCK"]``（2 小时），进程被杀时不会释放；
    Redis 又是独立容器，锁会跨重启存活，导致同一 URL 的新任务静默阻塞到锁自然过期。
    本进程刚启动、尚未处理任何任务，所以此刻残留的锁必然都是孤儿。

    注意：这假定单实例部署。若将来横向扩展，需改为按实例标识甄别持有者。
    """
    store = store or RedisCache()
    cleared = 0
    try:
        async for key in store.scan_iter(match=f"{UPLOAD_LOCK_PREFIX}*"):
            await store.delete(key if isinstance(key, str) else key.decode("utf-8", "ignore"))
            cleared += 1
    except Exception as err:
        # 清理只是启动优化，Redis 异常不该拖垮整个进程的启动
        logger.warning(f"清理遗留媒体处理锁失败，跳过: {type(err).__name__}: {err}")
        return cleared
    if cleared:
        logger.warning(f"清理了 {cleared} 个上次退出遗留的媒体处理锁")
    return cleared
