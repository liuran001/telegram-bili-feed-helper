"""测试 utils.py — logger、compress、escape_markdown、get_filename"""

import io

from PIL import Image

from biliparser.utils import compress, escape_markdown, get_filename, logger


def test_escape_markdown_special_chars():
    assert escape_markdown("hello_world") == r"hello\_world"
    assert escape_markdown("a*b") == r"a\*b"
    assert escape_markdown("[link](url)") == r"\[link\]\(url\)"


def test_escape_markdown_empty():
    assert escape_markdown("") == ""
    assert escape_markdown(None) == ""


def test_escape_markdown_html_entities():
    """html.unescape 先解码 &amp; -> &，& 不是 MarkdownV2 特殊字符所以不转义"""
    result = escape_markdown("a&amp;b")
    assert "&amp;" not in result
    assert result == "a&b"


def test_get_filename_normal():
    assert get_filename("https://example.com/path/image.jpg") == "image.jpg"
    assert get_filename("https://example.com/path/video.mp4?quality=high") == "video.mp4"


def test_get_filename_no_match():
    assert get_filename("no-extension-url") == "no-extension-url"


def test_compress_png():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    result = compress(buf, size=50)
    assert isinstance(result, io.BytesIO)
    result.seek(0)
    out = Image.open(result)
    assert max(out.size) <= 50


def test_compress_jpeg():
    img = Image.new("RGB", (200, 200), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    result = compress(buf, size=100, format="JPEG")
    result.seek(0)
    out = Image.open(result)
    assert out.mode == "RGB"


def test_compress_fix_ratio_wide():
    """超宽图应被 pad 到 1:20 比例"""
    img = Image.new("RGBA", (2100, 10), (0, 0, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    result = compress(buf, size=0, fix_ratio=True)
    result.seek(0)
    out = Image.open(result)
    w, h = out.size
    assert w / h <= 20


def test_compress_fix_ratio_tall():
    """超高图应被 pad 到 20:1 比例，且图像内容不能丢失（回归：之前 paste 偏移用错变量导致越界）"""
    img = Image.new("RGB", (1080, 25967), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    buf.seek(0)
    result = compress(buf, size=1280, fix_ratio=True)
    result.seek(0)
    out = Image.open(result)
    w, h = out.size
    assert h / w <= 20.01
    # 像素最大值 > 0 说明内容被正确粘贴进画布（越界 bug 会得到全黑/全透明图）
    assert out.convert("L").getextrema()[1] > 0


def test_logger_exists():
    assert logger is not None
