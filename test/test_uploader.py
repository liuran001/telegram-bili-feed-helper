"""测试共享下载层与 Telegram 上传队列的合并行为。"""

import asyncio
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from biliparser.channel.telegram.uploader import (
    _MEDIA_READ_TIMEOUT,
    TelegramUploadQueueManager,
    TelegramUploadTask,
    _cache_key,
    _send_kind,
    cache_media,
    get_cached_media_file_id,
)
from biliparser.model import Author, MediaConstraints, MediaInfo, ParsedContent
from biliparser.provider import ProviderRegistry
from biliparser.storage.models import TelegramFileCache
from biliparser.uploader.download import cleanup_medias, get_media_for_content, split_long_image_bytes


def _media_constraints() -> MediaConstraints:
    return MediaConstraints(
        max_upload_size=50 * 1024 * 1024,
        max_download_size=2 * 1024 * 1024 * 1024,
        caption_max_length=1024,
    )


def _manager(registry: ProviderRegistry | None = None) -> TelegramUploadQueueManager:
    return TelegramUploadQueueManager(
        registry=registry or ProviderRegistry(),
        constraints=_media_constraints(),
    )


def _make_jpeg(w: int, h: int) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), (120, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


def test_split_long_image_normal_returns_none():
    """正常比例图不切割，返回 None"""
    assert split_long_image_bytes(_make_jpeg(1080, 1080)) is None
    assert split_long_image_bytes(_make_jpeg(1080, 1920)) is None


def test_split_long_image_tall_splits_full_width():
    """超长图按高度切片，每片保留原始宽度且符合 tdlib 约束(w+h<=10000)"""
    pieces = split_long_image_bytes(_make_jpeg(1080, 15631), ratio=2, max_pieces=10)
    assert pieces is not None and len(pieces) > 1
    for pb in pieces:
        with Image.open(BytesIO(pb)) as im:
            w, h = im.size
        assert w == 1080  # 全宽，非细条
        assert w + h <= 10000  # tdlib photo 约束


def test_split_long_image_respects_max_pieces():
    """极端超长图切割片数不超过 max_pieces"""
    pieces = split_long_image_bytes(_make_jpeg(1080, 25967), ratio=2, max_pieces=10)
    assert pieces is not None and len(pieces) <= 10


def test_cleanup_medias_paths():
    """Path 类型的文件应被删除"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        p = Path(f.name)
        f.write(b"test")
    assert p.exists()
    cleanup_medias([p])
    assert not p.exists()


def test_cleanup_medias_strings():
    """字符串类型（file_id）不应被删除"""
    cleanup_medias(["file_id_123", "another_id"])  # 不应抛异常


def test_cleanup_medias_mixed():
    """混合类型应只删除 Path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        p = Path(f.name)
        f.write(b"test")
    cleanup_medias(["file_id", p, "another_id"])
    assert not p.exists()


def test_cleanup_medias_missing_file():
    """不存在的文件不应抛异常"""
    cleanup_medias([Path("/tmp/nonexistent_file_12345.jpg")])


def test_cleanup_medias_empty():
    cleanup_medias([])


def test_telegram_channel_constraints():
    from biliparser.channel.telegram import TelegramChannel

    mc = TelegramChannel().media_constraints
    assert mc.max_upload_size == 50 * 1024 * 1024  # 50MB (non-local mode)
    assert mc.max_download_size == 2 * 1024 * 1024 * 1024
    assert mc.caption_max_length == 1024


# ---- file_id 缓存按媒体类型命名空间隔离（防止 document file_id 被当 photo 用）----


def test_cache_key_namespaced_by_kind():
    """同一文件名在不同 kind 下生成不同主键"""
    fn = "abc.jpg"
    assert _cache_key(fn, "photo") != _cache_key(fn, "document")
    assert _cache_key(fn, "photo") == "photo:abc.jpg"
    assert _cache_key(fn, "document") == "document:abc.jpg"


def test_cache_key_truncated_to_primary_key_limit():
    """主键长度受 CharField(64) 限制，超长需截断而非溢出"""
    key = _cache_key("x" * 100, "document")
    assert len(key) <= 64


def test_send_kind_dispatch():
    """发送类型推断：/fetch document 优先，其次按 media.type 与 .gif 后缀"""
    assert _send_kind("image", "http://a.jpg", document=True) == "document"
    assert _send_kind("video", "http://a.mp4", document=True) == "document"  # fetch 总是 document
    assert _send_kind("video", "http://a.mp4") == "video"
    assert _send_kind("audio", "http://a.m4a") == "audio"
    assert _send_kind("image", "http://a.gif") == "gif"
    assert _send_kind("image", "http://a.jpg") == "photo"
    assert _send_kind("image", None) == "photo"


class _Att:
    def __init__(self, fid):
        self.file_id = fid


async def test_cache_does_not_cross_media_types(monkeypatch):
    """回归：同一文件名先以 document 缓存，photo 读取必须 miss（否则 document file_id 会被塞进 InputMediaPhoto）"""
    stored: dict[str, _Att] = {}

    async def update_or_create(*, mediafilename, defaults):
        stored[mediafilename] = _Att(defaults["file_id"])

    async def get_or_none(*, mediafilename):
        return stored.get(mediafilename)

    monkeypatch.setattr(TelegramFileCache, "update_or_create", update_or_create)
    monkeypatch.setattr(TelegramFileCache, "get_or_none", get_or_none)

    fn = "1211407624821538837_p1.jpg"
    await cache_media(fn, _Att("DOC_FILE_ID"), "document")

    # photo 命名空间下读取不应命中 document 的 file_id
    assert await get_cached_media_file_id(fn, "photo") is None
    # document 命名空间下能正确命中
    assert await get_cached_media_file_id(fn, "document") == "DOC_FILE_ID"

    # 之后以 photo 正确上传并缓存，两者互不覆盖
    await cache_media(fn, _Att("PHOTO_FILE_ID"), "photo")
    assert await get_cached_media_file_id(fn, "photo") == "PHOTO_FILE_ID"
    assert await get_cached_media_file_id(fn, "document") == "DOC_FILE_ID"


# ---- 动态图片只发前 10 张，不拆多组刷屏 ----


async def test_media_group_caps_at_ten_single_group():
    """超过 10 张图（含切片后）只发一个 media group，caption 挂在第一项作说明，不额外发文字消息"""
    n = 15
    f = ParsedContent(
        url="https://t.bilibili.com/123",
        author=Author(name="tester"),
        content="",
        media=MediaInfo(
            urls=[f"https://i0.hdslb.com/p{i}.jpg" for i in range(n)],
            type="image",
            filenames=[f"p{i}.jpg" for i in range(n)],
        ),
    )
    message = MagicMock()
    # reply_media_group 返回与入参等长的结果元组，模拟 Telegram 行为
    message.reply_media_group = AsyncMock(side_effect=lambda media, **kw: tuple(MagicMock() for _ in media))
    message.reply_text = AsyncMock()

    mgr = _manager()
    result = await mgr._upload_media_group(message, f, list(f.media.urls), None, "caption")

    # 只发一个 media group
    assert message.reply_media_group.call_count == 1
    # 组内恰好 10 项
    sent = message.reply_media_group.call_args.args[0]
    assert len(sent) == 10
    # caption 只挂在第一项，其余项无 caption（避免重复/多余说明）
    assert sent[0].caption == "caption"
    assert all(item.caption is None for item in sent[1:])
    # 不再额外发独立文字消息（说明已随相册显示在下方）
    assert message.reply_text.call_count == 0
    # 返回结果与发送项数一致（供缓存对齐）
    assert len(result) == 10
    # media group 同样放宽 read_timeout（与单图/视频一致，避免大相册误超时重发整组）
    assert message.reply_media_group.call_args.kwargs["read_timeout"] == _MEDIA_READ_TIMEOUT


# ---- 上传重试保留显式画质（/video <画质>），避免重复发送不同清晰度 ----


async def test_retry_parse_url_preserves_explicit_quality():
    """回归：首次上传超时重试时，重新解析必须透传原始 extra（如显式画质），
    否则会退回自动降档、发出与首次不同清晰度的第二份视频。"""
    f = ParsedContent(
        url="https://www.bilibili.com/video/av1?p=1",
        author=Author(name="tester"),
        media=MediaInfo(urls=["v", "a"], type="video", filenames=["x.mp4"]),
    )
    message = MagicMock()
    task = TelegramUploadTask(
        user_id=1,
        context=message,
        message=MagicMock(),
        parsed_content=f,
        media=[],
        mediathumb=None,
        urls=["https://www.bilibili.com/video/av1?p=1"],
        extra={"quality": "dolby"},
    )

    registry = MagicMock(spec=ProviderRegistry)
    registry.parse = AsyncMock(return_value=[f])
    mgr = _manager(registry)
    ok = await mgr._retry_parse_url(task)

    assert ok is True
    registry.parse.assert_awaited_once_with(
        [f.url],
        mgr.constraints,
        extra={"quality": "dolby"},
    )


# ---- 视频上传带封面 cover + 放宽 read_timeout（修复 HDR/杜比无封面 与 大视频误超时重发）----


async def test_video_upload_passes_cover_and_read_timeout():
    """video 发送须同时传 thumbnail 与 cover（部分客户端只认 cover 才显示封面），
    并使用放宽的 read_timeout（大视频服务端处理慢，默认 60s 会误判超时触发重发）。"""
    f = ParsedContent(
        url="https://www.bilibili.com/video/av1?p=1",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["merged.mp4"],
            type="video",
            filenames=["merged.mp4"],
            thumbnail="http://cover.jpg",
            dimension={"width": 1920, "height": 1080, "rotate": 0},
        ),
    )
    message = MagicMock()
    message.reply_video = AsyncMock(return_value=MagicMock(effective_attachment=_Att("VID")))
    task = TelegramUploadTask(
        user_id=1,
        context=message,
        message=message,
        parsed_content=f,
        media=["merged.mp4"],
        mediathumb="thumb.jpg",
        urls=["https://www.bilibili.com/video/av1?p=1"],
    )

    mgr = _manager()
    await mgr._upload_media(task)

    assert message.reply_video.call_count == 1
    kwargs = message.reply_video.call_args.kwargs
    assert kwargs["thumbnail"] == "thumb.jpg"
    assert kwargs["cover"] == "thumb.jpg"
    assert kwargs["read_timeout"] == _MEDIA_READ_TIMEOUT
    assert _MEDIA_READ_TIMEOUT >= 600


def test_telegram_upload_task_uses_context_message():
    from telegram import Message

    message = MagicMock(spec=Message)
    task = TelegramUploadTask(
        user_id=1,
        context=message,
        parsed_content=ParsedContent(url="https://example.com", author=Author()),
        media=[],
        mediathumb=None,
        urls=["https://example.com"],
    )

    assert task.message is message


async def test_telegram_upload_success_deletes_share_message(monkeypatch):
    manager = _manager()
    message = MagicMock()
    task = TelegramUploadTask(
        user_id=1,
        context=message,
        parsed_content=ParsedContent(url="https://example.com", author=Author()),
        media=[],
        mediathumb=None,
        urls=["https://example.com"],
    )
    upload_media = AsyncMock(return_value=object())
    delete_share_message = AsyncMock()
    monkeypatch.setattr(manager, "_upload_media", upload_media)
    monkeypatch.setattr(manager, "_try_delete_share_message", delete_share_message)

    await manager._do_upload(task)

    upload_media.assert_awaited_once_with(task)
    delete_share_message.assert_awaited_once_with(task)


async def test_stop_workers_cancels_active_upload(monkeypatch):
    manager = _manager()
    started = asyncio.Event()
    blocker = asyncio.Event()

    async def process_upload(_task):
        started.set()
        await blocker.wait()

    monkeypatch.setattr(manager, "_process_upload", process_upload)
    task = TelegramUploadTask(
        user_id=1,
        context=MagicMock(),
        parsed_content=ParsedContent(url="https://example.com/active", author=Author()),
        media=[],
        mediathumb=None,
        urls=["https://example.com/active"],
    )
    await manager.submit(task)
    await manager.start_workers()
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(manager.stop_workers(), timeout=1)

    assert all(worker.done() for worker in manager.workers)


async def test_cancel_user_tasks_counts_running_task_once():
    manager = _manager()
    running = asyncio.create_task(asyncio.Event().wait())
    task = TelegramUploadTask(
        user_id=1,
        context=MagicMock(),
        parsed_content=ParsedContent(url="https://example.com/running", author=Author()),
        media=[],
        mediathumb=None,
        urls=["https://example.com/running"],
    )
    manager.active_tasks[1] = {task.task_id: task}
    manager.processing_tasks[1] = {task.task_id: running}

    assert await manager.cancel_user_tasks(1) == 1
    assert 1 not in manager.active_tasks
    assert 1 not in manager.processing_tasks
    await asyncio.gather(running, return_exceptions=True)


async def test_empty_video_urls_fall_back_to_caption(monkeypatch):
    from biliparser.uploader import queue as queue_module

    content = ParsedContent(
        url="https://example.com/empty-video",
        author=Author(name="tester"),
        media=MediaInfo(urls=[], type="video", filenames=[]),
    )
    assert await get_media_for_content(content) == ([], None)

    class Lock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    cache = MagicMock()
    cache.lock.return_value = Lock()
    monkeypatch.setattr(queue_module, "RedisCache", lambda: cache)

    message = MagicMock()
    message.reply_text = AsyncMock()
    task = TelegramUploadTask(
        user_id=1,
        context=message,
        message=message,
        parsed_content=content,
        media=[],
        mediathumb=None,
        urls=[content.url],
    )

    assert await _manager()._try_upload_once(task, attempt=1, max_retries=1) is True
    message.reply_text.assert_awaited_once()
