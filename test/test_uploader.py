"""测试 channel/telegram/uploader.py — cleanup_medias、_get_constraints、split_long_image_bytes"""

import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from biliparser.channel.telegram.uploader import (
    _cache_key,
    _get_constraints,
    _send_kind,
    cache_media,
    cleanup_medias,
    get_cached_media_file_id,
    split_long_image_bytes,
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


def test_get_constraints_default():
    mc = _get_constraints()
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


@pytest.fixture
async def _db():
    from tortoise import Tortoise

    await Tortoise.init(db_url="sqlite://:memory:", modules={"models": ["biliparser.storage.models"]})
    await Tortoise.generate_schemas()
    try:
        yield
    finally:
        await Tortoise.close_connections()


async def test_cache_does_not_cross_media_types(_db):
    """回归：同一文件名先以 document 缓存，photo 读取必须 miss（否则 document file_id 会被塞进 InputMediaPhoto）"""
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
    from unittest.mock import AsyncMock, MagicMock

    from biliparser.channel.telegram.uploader import UploadQueueManager
    from biliparser.model import Author, MediaInfo, ParsedContent

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

    mgr = UploadQueueManager()
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
