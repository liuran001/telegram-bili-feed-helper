"""测试 storage 层 — RedisCache、FakeRedis、FakeLock、TelegramFileCache"""

import tempfile
from pathlib import Path

import pytest

from biliparser.storage.cache import (
    UPLOAD_LOCK_PREFIX,
    FakeLock,
    FakeRedis,
    RedisCache,
    clear_stale_upload_locks,
    upload_lock_key,
)
from biliparser.storage.models import TelegramFileCache


class TestFakeRedis:
    def _make_cache(self, tmpdir):
        cache = FakeRedis()
        cache.cache_file = Path(tmpdir) / "test_cache.json"
        cache.cache = {"__version": 2}
        return cache

    @pytest.mark.asyncio
    async def test_set_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.set("key1", "value1", ex=60)
            result = await cache.get("key1")
            assert result == "value1"

    @pytest.mark.asyncio
    async def test_set_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.set("key1", b"bytes_value")
            result = await cache.get("key1")
            assert result == "bytes_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            result = await cache.get("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_nx_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.set("nx_key", "first", nx=True)
            await cache.set("nx_key", "second", nx=True)
            result = await cache.get("nx_key")
            assert result == "first"

    @pytest.mark.asyncio
    async def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.set("del_key", "value")
            await cache.delete("del_key")
            result = await cache.get("del_key")
            assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """删除不存在的 key 不应报错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_expiry(self):
        """过期的 key 应返回 None"""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.set("exp_key", "value", ex=1)
            # 手动设置过期时间为过去
            cache.cache["exp_key"]["timeout"] = int(time.time()) - 10
            result = await cache.get("exp_key")
            assert result is None

    @pytest.mark.asyncio
    async def test_incr_expire_ttl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            assert await cache.incr("counter") == 1
            assert await cache.incr("counter") == 2
            assert await cache.get("counter") == "2"
            assert await cache.ttl("counter") == -1
            assert await cache.expire("counter", 60) is True
            assert 0 < await cache.ttl("counter") <= 60

    @pytest.mark.asyncio
    async def test_version_key_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            result = await cache.get("__version")
            assert result is None

    def test_lock_returns_fake_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            lock = cache.lock("test_lock", timeout=10)
            assert isinstance(lock, FakeLock)

    @pytest.mark.asyncio
    async def test_persistence(self):
        """写入后文件应存在"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make_cache(tmpdir)
            await cache.set("persist_key", "persist_value")
            assert cache.cache_file.exists()


class TestFakeLock:
    @pytest.mark.asyncio
    async def test_acquire_release(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FakeRedis()
            store.cache_file = Path(tmpdir) / "lock_cache.json"
            store.cache = {"__version": 2}
            lock = FakeLock(store, "test_lock", timeout=10)
            acquired = await lock.acquire()
            assert acquired is True
            await lock.release()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FakeRedis()
            store.cache_file = Path(tmpdir) / "lock_cache.json"
            store.cache = {"__version": 2}
            lock = FakeLock(store, "ctx_lock", timeout=10)
            async with lock:
                pass  # 不应抛异常


class TestRedisCacheSingleton:
    def test_singleton(self):
        """RedisCache 应返回同一实例"""
        a = RedisCache()
        b = RedisCache()
        assert a is b


class TestTelegramFileCache:
    def test_model_table(self):
        assert TelegramFileCache._meta.db_table is not None

    def test_model_fields(self):
        field_names = {f for f in TelegramFileCache._meta.fields_map}
        assert "mediafilename" in field_names
        assert "file_id" in field_names
        assert "created" in field_names


class TestStaleUploadLockCleanup:
    """进程被杀后遗留的下载锁 — 锁 TTL 长达 2 小时，不清理会让同 URL 的新任务一直阻塞"""

    def test_lock_key_is_namespaced(self):
        """锁 key 需带前缀，才能与同库的缓存 key 区分开、安全批量清理"""
        key = upload_lock_key("https://www.bilibili.com/video/av123?p=1")
        assert key.startswith(UPLOAD_LOCK_PREFIX)
        assert key.endswith("https://www.bilibili.com/video/av123?p=1")

    async def test_clears_only_stale_upload_locks(self, tmp_path, monkeypatch):
        monkeypatch.setattr("biliparser.storage.cache.LOCAL_FILE_PATH", tmp_path)
        store = FakeRedis()
        store.cache = {}
        await store.set(upload_lock_key("https://www.bilibili.com/video/av1"), "1")
        await store.set(upload_lock_key("https://www.bilibili.com/video/av2"), "1")
        await store.set("video:aid:12345", "payload")
        await store.set("new_reply:12345:1", "payload")

        cleared = await clear_stale_upload_locks(store)

        assert cleared == 2
        assert await store.get("video:aid:12345") == "payload"
        assert await store.get("new_reply:12345:1") == "payload"
        assert await store.get(upload_lock_key("https://www.bilibili.com/video/av1")) is None
        assert await store.get(upload_lock_key("https://www.bilibili.com/video/av2")) is None

    async def test_no_locks_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("biliparser.storage.cache.LOCAL_FILE_PATH", tmp_path)
        store = FakeRedis()
        store.cache = {}
        await store.set("video:aid:1", "x")
        assert await clear_stale_upload_locks(store) == 0
        assert await store.get("video:aid:1") == "x"

    async def test_released_lock_leaves_nothing_to_clear(self, tmp_path, monkeypatch):
        """正常释放的锁不该在启动时被统计为孤儿"""
        monkeypatch.setattr("biliparser.storage.cache.LOCAL_FILE_PATH", tmp_path)
        store = FakeRedis()
        store.cache = {}
        async with store.lock(upload_lock_key("https://www.bilibili.com/video/av9"), timeout=60):
            assert await clear_stale_upload_locks(store) == 1  # 持锁期间可见
        assert await clear_stale_upload_locks(store) == 0

    async def test_cleanup_failure_does_not_block_startup(self):
        """清理挂在 db_init 里，Redis 异常绝不能拖垮启动"""

        class BrokenStore:
            def scan_iter(self, match):
                raise ConnectionError("redis down")

        assert await clear_stale_upload_locks(BrokenStore()) == 0

    async def test_cleanup_handles_bytes_keys(self, tmp_path, monkeypatch):
        """真实 redis-py 不开 decode_responses 时 scan_iter 产出 bytes"""
        monkeypatch.setattr("biliparser.storage.cache.LOCAL_FILE_PATH", tmp_path)
        deleted: list = []

        class BytesStore:
            async def scan_iter(self, match):
                yield upload_lock_key("https://www.bilibili.com/video/av1").encode()

            async def delete(self, key):
                deleted.append(key)

        assert await clear_stale_upload_locks(BytesStore()) == 1
        assert deleted == [upload_lock_key("https://www.bilibili.com/video/av1")]
