"""测试共享下载层与 Telegram 上传队列的合并行为。"""

import asyncio
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
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
from biliparser.uploader.download import (
    FILE_READ_TIMEOUT,
    cleanup_medias,
    get_media,
    get_media_for_content,
    get_media_from_candidates,
    normalize_media_url,
    rank_media_urls,
    split_long_image_bytes,
)


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


class _AsyncBytesStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        yield self.data


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


def test_bilibili_cdn_http_url_is_upgraded_to_https():
    assert normalize_media_url("http://i2.hdslb.com/bfs/archive/cover.jpg") == (
        "https://i2.hdslb.com/bfs/archive/cover.jpg"
    )
    assert normalize_media_url("http://example.com/file.jpg") == "http://example.com/file.jpg"


async def test_required_media_download_propagates_httpx_timeout():
    async def raise_timeout(request):
        raise httpx.ReadTimeout("slow CDN", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(raise_timeout)) as client:
        with pytest.raises(httpx.ReadTimeout):
            await get_media(
                client,
                "https://www.bilibili.com/video/BV1test",
                "https://cdn.example.com/video.m4s",
                "video.m4s",
                raise_on_error=True,
            )


async def test_upos_candidates_are_ranked_by_parallel_range_probe():
    async def probe(request):
        assert request.headers["Range"] == "bytes=0-1023"
        await asyncio.sleep(0.005 if request.url.host == "fast.example" else 0.05)
        return httpx.Response(206, stream=_AsyncBytesStream(b"x" * 1024))

    async with httpx.AsyncClient(transport=httpx.MockTransport(probe)) as client:
        ranked = await rank_media_urls(
            client,
            "https://www.bilibili.com/video/BV1test",
            ["https://slow.example/video.m4s", "https://fast.example/video.m4s"],
            probe_bytes=1024,
            probe_timeout=1,
        )

    assert ranked == ["https://fast.example/video.m4s", "https://slow.example/video.m4s"]


async def test_shared_semaphore_caps_parallel_upos_probes():
    active = 0
    max_active = 0

    async def probe(request):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            return httpx.Response(206, stream=_AsyncBytesStream(b"x" * 16))
        finally:
            active -= 1

    semaphore = asyncio.Semaphore(2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(probe)) as client:
        await asyncio.gather(
            rank_media_urls(
                client,
                "https://www.bilibili.com/video/BV1test",
                ["https://v1.example/v.m4s", "https://v2.example/v.m4s"],
                probe_bytes=16,
                probe_timeout=1,
                semaphore=semaphore,
            ),
            rank_media_urls(
                client,
                "https://www.bilibili.com/video/BV1test",
                ["https://a1.example/a.m4s", "https://a2.example/a.m4s"],
                probe_bytes=16,
                probe_timeout=1,
                semaphore=semaphore,
            ),
        )

    assert max_active == 2


async def test_upos_download_switches_to_next_candidate_after_failure(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    slow = "https://slow.example/video.m4s"
    fast = "https://fast.example/video.m4s"
    attempts = []
    expected = tmp_path / "video.m4s"

    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=[slow, fast]))

    async def download_candidate(client, referer, url, filename, **kwargs):
        attempts.append(url)
        assert kwargs["raise_on_error"] is True
        if url == slow:
            raise httpx.ConnectTimeout("slow UPOS")
        return expected

    monkeypatch.setattr(download_module, "get_media", download_candidate)

    result = await get_media_from_candidates(
        MagicMock(),
        "https://www.bilibili.com/video/BV1test",
        [slow, fast],
        "video.m4s",
    )

    assert result == expected
    assert attempts == [slow, fast]


async def test_large_media_uses_parallel_range_chunks_and_cross_upos_retry(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    backup_payload = payload.upper()
    urls = ["https://primary.example/video.m4s", "https://backup.example/video.m4s"]
    active_chunks = 0
    max_active_chunks = 0
    requests = []

    async def range_server(request):
        nonlocal active_chunks, max_active_chunks
        start, end = (int(value) for value in request.headers["Range"].removeprefix("bytes=").split("-"))
        if (start, end) == (0, 0):
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Type": "video/mp4",
                    "ETag": '"same-resource"',
                },
                stream=_AsyncBytesStream(payload[:1]),
            )

        requests.append((request.url.host, start, end))
        active_chunks += 1
        max_active_chunks = max(max_active_chunks, active_chunks)
        try:
            await asyncio.sleep(0.01)
            if request.url.host == "primary.example" and start == 0:
                return httpx.Response(503, stream=_AsyncBytesStream(b""))
            source_payload = backup_payload if request.url.host == "backup.example" else payload
            data = source_payload[start : end + 1]
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "Content-Type": "video/mp4",
                },
                stream=_AsyncBytesStream(data),
            )
        finally:
            active_chunks -= 1

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 3)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 8)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_RETRIES", 2)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNKS_PER_WORKER", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_HEDGE_DELAY", 0.005)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=urls))

    async with httpx.AsyncClient(transport=httpx.MockTransport(range_server)) as client:
        result = await get_media_from_candidates(
            client,
            "https://www.bilibili.com/video/BV1test",
            urls,
            "video.m4s",
            raise_on_error=True,
            parallel_ranges=True,
        )

    assert isinstance(result, Path)
    assert result.read_bytes() == backup_payload
    assert max_active_chunks >= 2
    assert ("primary.example", 0, 7) in requests
    assert ("backup.example", 0, 7) in requests


async def test_range_download_adaptively_reduces_concurrency(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    url = "https://primary.example/video.m4s"

    async def range_server(request):
        start, end = (int(value) for value in request.headers["Range"].removeprefix("bytes=").split("-"))
        if (start, end) == (0, 0):
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Type": "video/mp4",
                    "ETag": '"stable-resource"',
                },
                stream=_AsyncBytesStream(payload[:1]),
            )
        return httpx.Response(
            206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
            stream=_AsyncBytesStream(payload[start : end + 1]),
        )

    original_download = download_module._download_ranges_from_source
    worker_levels = []

    async def fail_high_concurrency(client, referer, source, filename, semaphore, workers):
        worker_levels.append(workers)
        if workers > 1:
            raise download_module._RangeTransferError("simulated CDN concurrency limit")
        return await original_download(client, referer, source, filename, semaphore, workers)

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 4)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 8)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNKS_PER_WORKER", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_HEDGE_DELAY", 0)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=[url]))
    monkeypatch.setattr(download_module, "_download_ranges_from_source", fail_high_concurrency)

    async with httpx.AsyncClient(transport=httpx.MockTransport(range_server)) as client:
        result = await get_media_from_candidates(
            client,
            "https://www.bilibili.com/video/BV1test",
            [url],
            "video.m4s",
            raise_on_error=True,
            parallel_ranges=True,
        )

    assert isinstance(result, Path)
    assert result.read_bytes() == payload
    assert worker_levels == [4, 2, 1]


async def test_range_unsupported_falls_back_to_single_stream(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    urls = ["https://primary.example/video.m4s", "https://backup.example/video.m4s"]
    expected = tmp_path / "video.m4s"
    fallback = AsyncMock(return_value=expected)

    async def no_range(request):
        return httpx.Response(200, stream=_AsyncBytesStream(b"not-a-range-response"))

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 4)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=urls))
    monkeypatch.setattr(download_module, "get_media", fallback)

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_range)) as client:
        result = await get_media_from_candidates(
            client,
            "https://www.bilibili.com/video/BV1test",
            urls,
            "video.m4s",
            parallel_ranges=True,
        )

    assert result == expected
    fallback.assert_awaited_once()


@pytest.mark.parametrize("etag", ["", 'W/"weak"', 'w/"invalid"', "unquoted", '"one", "two"'])
async def test_range_without_strong_etag_falls_back_to_single_stream(monkeypatch, tmp_path, etag):
    from biliparser.uploader import download as download_module

    url = "https://primary.example/video.m4s"
    expected = tmp_path / "video.m4s"
    fallback = AsyncMock(return_value=expected)

    async def no_validator(request):
        return httpx.Response(
            206,
            headers={"Content-Range": "bytes 0-0/64", "Content-Type": "video/mp4", "ETag": etag},
            stream=_AsyncBytesStream(b"x"),
        )

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 4)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=[url]))
    monkeypatch.setattr(download_module, "get_media", fallback)

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_validator)) as client:
        result = await get_media_from_candidates(
            client,
            "https://www.bilibili.com/video/BV1test",
            [url],
            "video.m4s",
            parallel_ranges=True,
        )

    assert result == expected
    fallback.assert_awaited_once()


async def test_range_advertised_size_over_limit_does_not_preallocate(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    url = "https://primary.example/video.m4s"
    expected = tmp_path / "video.m4s"
    fallback = AsyncMock(return_value=expected)

    async def huge_resource(request):
        total = 10_000
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes 0-0/{total}",
                "Content-Type": "video/mp4",
                "ETag": '"huge-resource"',
            },
            stream=_AsyncBytesStream(b"x"),
        )

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 4)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_MAX_SIZE", 9_999)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=[url]))
    monkeypatch.setattr(download_module, "get_media", fallback)

    async with httpx.AsyncClient(transport=httpx.MockTransport(huge_resource)) as client:
        result = await get_media_from_candidates(
            client,
            "https://www.bilibili.com/video/BV1test",
            [url],
            "video.m4s",
            parallel_ranges=True,
        )

    assert result == expected
    assert list(tmp_path.iterdir()) == []


async def test_range_local_io_error_is_not_masked_by_single_stream_fallback(monkeypatch):
    from biliparser.uploader import download as download_module

    url = "https://primary.example/video.m4s"
    fallback = AsyncMock()
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=[url]))
    monkeypatch.setattr(download_module, "get_media_by_ranges", AsyncMock(side_effect=OSError("disk full")))
    monkeypatch.setattr(download_module, "get_media", fallback)

    with pytest.raises(OSError, match="disk full"):
        await get_media_from_candidates(
            MagicMock(),
            "https://www.bilibili.com/video/BV1test",
            [url],
            "video.m4s",
            parallel_ranges=True,
        )

    fallback.assert_not_awaited()


async def test_range_download_cancellation_removes_partial_file(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    urls = ["https://primary.example/video.m4s"]
    chunk_started = asyncio.Event()
    never = asyncio.Event()

    async def range_server(request):
        start, end = (int(value) for value in request.headers["Range"].removeprefix("bytes=").split("-"))
        if (start, end) == (0, 0):
            return httpx.Response(
                206,
                headers={
                    "Content-Range": "bytes 0-0/64",
                    "Content-Type": "video/mp4",
                    "ETag": '"stable-resource"',
                },
                stream=_AsyncBytesStream(b"x"),
            )
        chunk_started.set()
        await never.wait()

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 2)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 8)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNKS_PER_WORKER", 1)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=urls))

    async with httpx.AsyncClient(transport=httpx.MockTransport(range_server)) as client:
        running = asyncio.create_task(
            get_media_from_candidates(
                client,
                "https://www.bilibili.com/video/BV1test",
                urls,
                "video.m4s",
                raise_on_error=True,
                parallel_ranges=True,
            )
        )
        await asyncio.wait_for(chunk_started.wait(), timeout=1)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    assert running.cancelled()
    assert list(tmp_path.iterdir()) == []


async def test_tail_slow_chunk_is_hedged_on_spare_connection(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    payload = b"abcdefghijklmnopqrstuvwx"
    urls = ["https://primary.example/video.m4s", "https://backup.example/video.m4s"]
    requests = []
    tail_primary_requests = 0

    async def range_server(request):
        nonlocal tail_primary_requests
        start, end = (int(value) for value in request.headers["Range"].removeprefix("bytes=").split("-"))
        if (start, end) == (0, 0):
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Type": "video/mp4",
                    "ETag": '"same-resource"',
                },
                stream=_AsyncBytesStream(payload[:1]),
            )
        requests.append((request.url.host, start, end))
        if start == 16 and request.url.host == "primary.example":
            tail_primary_requests += 1
            await asyncio.sleep(0.1 if tail_primary_requests == 1 else 0.001)
        else:
            await asyncio.sleep(0.001)
        return httpx.Response(
            206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
            stream=_AsyncBytesStream(payload[start : end + 1]),
        )

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "FILE_RANGE_WORKERS", 2)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 8)
    monkeypatch.setattr(download_module, "FILE_RANGE_MIN_SIZE", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_RETRIES", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNKS_PER_WORKER", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_HEDGE_DELAY", 0.005)
    monkeypatch.setattr(download_module, "rank_media_urls", AsyncMock(return_value=urls))

    loop = asyncio.get_running_loop()
    started = loop.time()
    async with httpx.AsyncClient(transport=httpx.MockTransport(range_server)) as client:
        result = await get_media_from_candidates(
            client,
            "https://www.bilibili.com/video/BV1test",
            urls,
            "video.m4s",
            raise_on_error=True,
            parallel_ranges=True,
        )
    elapsed = loop.time() - started

    assert isinstance(result, Path)
    assert result.read_bytes() == payload
    assert requests.count(("primary.example", 16, 23)) >= 2
    assert elapsed < 0.08


def test_range_chunk_plan_oversegments_to_reduce_tail_slowdown(monkeypatch):
    from biliparser.uploader import download as download_module

    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 10_000)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNKS_PER_WORKER", 8)

    chunks = download_module._build_range_chunks(total=10_000, workers=4)

    assert len(chunks) >= 32
    assert max(end - start + 1 for _, start, end in chunks) <= 313


def test_range_chunk_plan_is_bounded_for_tiny_configured_chunks(monkeypatch):
    from biliparser.uploader import download as download_module

    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 1)
    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNKS_PER_WORKER", 1_000_000)
    monkeypatch.setattr(download_module, "FILE_RANGE_MAX_CHUNKS", 128)

    chunks = download_module._build_range_chunks(total=10_000, workers=4)

    assert len(chunks) <= 128
    assert chunks[0][1] == 0
    assert chunks[-1][2] == 9_999


def test_range_chunk_plan_rejects_conflicting_hard_limits(monkeypatch):
    from biliparser.uploader import download as download_module

    monkeypatch.setattr(download_module, "FILE_RANGE_CHUNK_SIZE", 512 * 1024 * 1024)
    monkeypatch.setattr(download_module, "FILE_RANGE_MAX_CHUNK_SIZE", 64 * 1024 * 1024)
    monkeypatch.setattr(download_module, "FILE_RANGE_MAX_CHUNKS", 4)

    chunks = download_module._build_range_chunks(total=2 * 1024 * 1024 * 1024, workers=4)

    assert chunks == []


async def test_dash_failure_uses_mp4_fallback(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    content = ParsedContent(
        url="https://www.bilibili.com/video/BV1test",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["https://cdn.example.com/video.m4s", "https://cdn.example.com/audio.m4s"],
            type="video",
            filenames=["video.m4s", "audio.m4s"],
            need_download=True,
            merge_streams=True,
            fallback_url="https://cdn.example.com/fallback.mp4",
            fallback_candidates=[
                "https://cdn.example.com/fallback.mp4",
                "https://backup.example.com/fallback.mp4",
            ],
        ),
    )
    fallback_path = tmp_path / "video_merged.mp4"

    async def fail_dash(*args, **kwargs):
        raise httpx.ReadTimeout("slow DASH")

    async def download_fallback(client, referer, urls, filename, **kwargs):
        assert urls == content.media.fallback_candidates
        assert filename == "video_merged.mp4"
        assert kwargs["raise_on_error"] is True
        return fallback_path

    monkeypatch.setattr(download_module, "handle_dash_media", fail_dash)
    monkeypatch.setattr(download_module, "get_media_from_candidates", download_fallback)

    media, thumbnail = await get_media_for_content(content)

    assert media == [fallback_path]
    assert thumbnail is None
    assert content.media.urls == [content.media.fallback_url]
    assert content.media.filenames == ["video_merged.mp4"]
    assert content.media.merge_streams is False


async def test_dash_download_uses_candidates_for_each_track(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    video_candidates = ["https://video-primary.example/v.m4s", "https://video-backup.example/v.m4s"]
    audio_candidates = ["https://audio-primary.example/a.m4s", "https://audio-backup.example/a.m4s"]
    content = ParsedContent(
        url="https://www.bilibili.com/video/BV1test",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=[video_candidates[0], audio_candidates[0]],
            url_candidates=[video_candidates, audio_candidates],
            type="video",
            filenames=["video.m4s", "audio.m4s"],
            need_download=True,
            merge_streams=True,
        ),
    )
    calls = {}

    async def download_candidates(client, referer, urls, filename, **kwargs):
        calls[filename] = urls
        path = tmp_path / filename
        path.write_bytes(b"track")
        return path

    def merge(cmd, check):
        assert check is True
        Path(cmd[-1]).write_bytes(b"merged")

    monkeypatch.setattr(download_module, "LOCAL_MEDIA_FILE_PATH", tmp_path)
    monkeypatch.setattr(download_module, "get_media_from_candidates", download_candidates)
    monkeypatch.setattr(download_module.subprocess, "run", merge)

    media, thumbnail = await get_media_for_content(content)

    assert thumbnail is None
    assert media == [tmp_path / "video_merged.mp4"]
    assert calls == {"video.m4s": video_candidates, "audio.m4s": audio_candidates}
    assert content.media.url_candidates == []


async def test_dash_cancellation_cleans_track_that_already_finished(monkeypatch, tmp_path):
    from biliparser.uploader import download as download_module

    content = ParsedContent(
        url="https://www.bilibili.com/video/BV1test",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["https://video.example/v.m4s", "https://audio.example/a.m4s"],
            type="video",
            filenames=["video.m4s", "audio.m4s"],
            need_download=True,
            merge_streams=True,
        ),
    )
    audio_path = tmp_path / "audio.m4s"
    audio_finished = asyncio.Event()
    video_started = asyncio.Event()
    never = asyncio.Event()

    async def download_track(client, referer, urls, filename, **kwargs):
        if filename == "audio.m4s":
            audio_path.write_bytes(b"audio")
            audio_finished.set()
            return audio_path
        video_started.set()
        await never.wait()

    monkeypatch.setattr(download_module, "get_media_from_candidates", download_track)

    running = asyncio.create_task(download_module.handle_dash_media(content, MagicMock()))
    await asyncio.wait_for(asyncio.gather(audio_finished.wait(), video_started.wait()), timeout=1)
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)

    assert running.cancelled()
    assert not audio_path.exists()


async def test_dash_failure_without_fallback_propagates(monkeypatch):
    from biliparser.uploader import download as download_module

    content = ParsedContent(
        url="https://www.bilibili.com/video/BV1test",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["https://cdn.example.com/video.m4s", "https://cdn.example.com/audio.m4s"],
            type="video",
            filenames=["video.m4s", "audio.m4s"],
            need_download=True,
            merge_streams=True,
        ),
    )

    async def fail_dash(*args, **kwargs):
        raise httpx.ReadTimeout("slow DASH")

    monkeypatch.setattr(download_module, "handle_dash_media", fail_dash)

    with pytest.raises(httpx.ReadTimeout):
        await get_media_for_content(content)


def test_file_read_timeout_is_not_httpx_five_second_default():
    assert FILE_READ_TIMEOUT >= 60


async def test_media_client_receives_configured_timeout(monkeypatch):
    from biliparser.uploader import download as download_module

    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(download_module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(download_module, "LOCAL_MODE", False)
    content = ParsedContent(
        url="https://example.com/post",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["https://cdn.example.com/image.jpg"],
            type="image",
            filenames=["image.jpg"],
        ),
    )

    await get_media_for_content(content)

    assert captured["timeout"].read == FILE_READ_TIMEOUT
    assert captured["http2"] is False


async def test_fetch_image_download_failure_propagates_for_queue_retry(monkeypatch):
    """``/fetch`` 下载原文件失败时必须抛出网络异常，不能过滤成空媒体列表。"""
    from biliparser.uploader import download as download_module

    content = ParsedContent(
        url="https://t.bilibili.com/123",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["https://i0.hdslb.com/bfs/new_dyn/image.png"],
            type="image",
            filenames=["image.png"],
            need_download=True,
        ),
    )

    async def fail_download(*args, raise_on_error=False, **kwargs):
        assert raise_on_error is True
        raise httpx.ConnectTimeout("slow CDN")

    monkeypatch.setattr(download_module, "get_media", fail_download)

    with pytest.raises(httpx.ConnectTimeout):
        await get_media_for_content(content, media_check_ignore=True)


async def test_fetch_never_sends_empty_media_group(monkeypatch):
    """下载层意外返回空结果时，fetch 应抛出可重试错误而非调用 media=[]。"""
    from biliparser.channel.telegram import uploader as uploader_module

    class Lock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    cache = MagicMock()
    cache.lock.return_value = Lock()
    monkeypatch.setattr(uploader_module, "RedisCache", lambda: cache)
    monkeypatch.setattr(uploader_module, "get_media_for_content", AsyncMock(return_value=([], None)))

    content = ParsedContent(
        url="https://t.bilibili.com/123",
        author=Author(name="tester"),
        media=MediaInfo(
            urls=["https://i0.hdslb.com/bfs/new_dyn/image.png"],
            type="image",
            filenames=["image.png"],
            need_download=True,
        ),
    )
    message = MagicMock()
    message.reply_media_group = AsyncMock()
    task = TelegramUploadTask(
        user_id=1,
        context=message,
        message=message,
        parsed_content=content,
        media=[],
        mediathumb=None,
        urls=[content.url],
        task_type="fetch",
        fetch_mode="file",
    )

    with pytest.raises(httpx.RequestError, match="未下载到可发送的媒体"):
        await _manager()._process_fetch_task(task)

    message.reply_media_group.assert_not_awaited()


def test_exception_logging_does_not_render_sensitive_object_repr(tmp_path):
    """生产异常栈不应通过 Loguru 变量诊断展开 Bot token 等对象内容。"""
    sentinel = "SECRET_TOKEN_SENTINEL_7f9a"
    script = tmp_path / "loguru_diagnose_probe.py"
    script.write_text(
        f'''from biliparser.utils import logger

class SensitiveObject:
    def __repr__(self):
        return "{sentinel}"

def crash(value):
    return value.missing_attribute

try:
    crash(SensitiveObject())
except AttributeError:
    logger.exception("expected test exception")
''',
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )

    assert sentinel not in result.stdout + result.stderr


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
