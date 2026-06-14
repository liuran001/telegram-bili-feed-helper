"""测试 channel/telegram/uploader.py — cleanup_medias、_get_constraints、split_long_image_bytes"""

import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from biliparser.channel.telegram.uploader import (
    _get_constraints,
    cleanup_medias,
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
