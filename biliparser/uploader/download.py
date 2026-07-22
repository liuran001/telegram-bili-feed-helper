"""
平台无关的媒体下载逻辑

提供给所有 channel 复用：get_media, handle_dash_media, get_media_for_content, cleanup_medias

cache_lookup: 可选的缓存查询函数，签名为 async (filename: str) -> str | None
  由调用方注入（如 Telegram channel 注入 TelegramFileCache 查询）
  不传则跳过缓存查询，始终重新下载
"""

import asyncio
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from async_timeout import timeout
from PIL import Image
from tqdm import tqdm

from ..model import ParsedContent
from ..provider.bilibili.api import BILIBILI_DESKTOP_HEADER, CACHES_TIMER, referer_url
from ..utils import compress, logger

LOCAL_MEDIA_FILE_PATH = Path(os.environ.get("LOCAL_TEMP_FILE_PATH", str(Path.cwd()))) / ".tmp"
LOCAL_MODE = bool(os.environ.get("LOCAL_MODE", False))
IMAGE_SPLIT_RATIO = float(os.environ.get("IMAGE_SPLIT_RATIO", 2))
IMAGE_SPLIT_MAX_PIECES = int(os.environ.get("IMAGE_SPLIT_MAX_PIECES", 10))
FILE_CONNECT_TIMEOUT = float(os.environ.get("FILE_CONNECT_TIMEOUT", 30))
FILE_READ_TIMEOUT = float(os.environ.get("FILE_READ_TIMEOUT", 60))
FILE_WRITE_TIMEOUT = float(os.environ.get("FILE_WRITE_TIMEOUT", 60))
FILE_POOL_TIMEOUT = float(os.environ.get("FILE_POOL_TIMEOUT", 30))
UPOS_PROBE_BYTES = int(os.environ.get("UPOS_PROBE_BYTES", 2 * 1024 * 1024))
UPOS_PROBE_TIMEOUT = float(os.environ.get("UPOS_PROBE_TIMEOUT", 10))
DOWNLOAD_PROGRESS_INTERVAL = float(os.environ.get("DOWNLOAD_PROGRESS_INTERVAL", 15))
FILE_RANGE_WORKERS = int(os.environ.get("FILE_RANGE_WORKERS", 4))
FILE_RANGE_MAX_WORKERS = int(os.environ.get("FILE_RANGE_MAX_WORKERS", 16))
FILE_RANGE_CHUNK_SIZE = int(os.environ.get("FILE_RANGE_CHUNK_SIZE", 8 * 1024 * 1024))
FILE_RANGE_MAX_CHUNK_SIZE = int(os.environ.get("FILE_RANGE_MAX_CHUNK_SIZE", 64 * 1024 * 1024))
FILE_RANGE_MIN_SIZE = int(os.environ.get("FILE_RANGE_MIN_SIZE", 32 * 1024 * 1024))
FILE_RANGE_RETRIES = int(os.environ.get("FILE_RANGE_RETRIES", 2))
FILE_RANGE_CHUNKS_PER_WORKER = int(os.environ.get("FILE_RANGE_CHUNKS_PER_WORKER", 16))
FILE_RANGE_MAX_CHUNKS = int(os.environ.get("FILE_RANGE_MAX_CHUNKS", 4096))
FILE_RANGE_MAX_SIZE = int(os.environ.get("FILE_RANGE_MAX_SIZE", 2 * 1024 * 1024 * 1024))
FILE_RANGE_REQUEST_TIMEOUT = float(os.environ.get("FILE_RANGE_REQUEST_TIMEOUT", 30))
FILE_RANGE_HEDGE_DELAY = float(os.environ.get("FILE_RANGE_HEDGE_DELAY", 15))

CacheLookup = Callable[[str], Coroutine[None, None, str | None]]
CacheKeyBuilder = Callable[[str, str | None, Path | str | None, bool], str]
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_STRONG_ETAG_RE = re.compile(r'^"[\x21\x23-\x7e\x80-\xff]*"$')


@dataclass(frozen=True)
class _RangeSource:
    url: str
    total: int
    content_type: str
    etag: str = ""

    @property
    def strong_etag(self) -> str:
        etag = self.etag.strip()
        return etag if _STRONG_ETAG_RE.fullmatch(etag) else ""


class _RangeTransferError(Exception):
    """Range 能力已确认，但分块传输未能完整、安全地完成。"""


def normalize_media_url(url: str) -> str:
    """Bilibili CDN 的 HTTP 链接统一升级为 HTTPS，避免部分网络环境直连 80 端口超时。"""
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if parsed.scheme == "http" and any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in ("hdslb.com", "bilivideo.com")
    ):
        return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return url


def _media_url_host(url: str) -> str:
    return urlsplit(url).hostname or url


def _display_media_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _dedupe_media_urls(urls: list[str]) -> list[str]:
    return list(dict.fromkeys(normalize_media_url(url) for url in urls if url))


def _range_worker_count() -> int:
    return min(max(1, FILE_RANGE_WORKERS), max(1, FILE_RANGE_MAX_WORKERS))


def cleanup_medias(medias) -> None:
    """删除临时下载的媒体文件（Path 类型），跳过字符串（file_id）"""
    for item in medias:
        if isinstance(item, Path):
            item.unlink(missing_ok=True)


async def get_media(
    client: httpx.AsyncClient,
    referer,
    url: Path | str,
    filename: str,
    compression: bool = True,
    media_check_ignore: bool = False,
    no_cache: bool = False,
    is_thumbnail: bool = False,
    cache_lookup: CacheLookup | None = None,
    cache_key: str | None = None,
    raise_on_error: bool = False,
) -> Path | str | list[Path] | None:
    """下载单个媒体文件到本地临时目录，返回本地 Path 或缓存 file_id"""
    if isinstance(url, Path):
        return url
    url = normalize_media_url(url)
    if not no_cache and cache_lookup is not None:
        file_id = await cache_lookup(cache_key or filename)
        if file_id:
            return file_id
    LOCAL_MEDIA_FILE_PATH.mkdir(parents=True, exist_ok=True)
    media = LOCAL_MEDIA_FILE_PATH / filename
    temp_media = LOCAL_MEDIA_FILE_PATH / uuid4().hex
    try:
        header = BILIBILI_DESKTOP_HEADER.copy()
        header["Referer"] = referer
        async with timeout(CACHES_TIMER["LOCK"]), client.stream("GET", url, headers=header) as response:
            logger.info(f"下载开始: {_display_media_url(url)} -> {filename}")
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"媒体文件获取错误: {response.status_code} {url}->{referer}",
                    request=response.request,
                    response=response,
                )
            content_type = response.headers.get("content-type")
            if content_type is None:
                raise httpx.HTTPStatusError(
                    f"媒体文件获取错误: 无法获取 content-type {url}->{referer}",
                    request=response.request,
                    response=response,
                )
            mediatype = content_type.split("/")
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            download_started = time.monotonic()
            last_progress = download_started
            if mediatype[0] in ["video", "audio", "application"]:
                with (
                    temp_media.open("wb") as file,
                    tqdm(
                        total=total,
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                        desc=response.request.url.host + "->" + filename,
                        disable=not sys.stderr.isatty(),
                    ) as pbar,
                ):
                    async for chunk in response.aiter_bytes():
                        file.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        now = time.monotonic()
                        if DOWNLOAD_PROGRESS_INTERVAL > 0 and now - last_progress >= DOWNLOAD_PROGRESS_INTERVAL:
                            elapsed = max(now - download_started, 0.001)
                            speed = downloaded / elapsed / (1024 * 1024)
                            progress = f"{downloaded / (1024 * 1024):.1f} MiB"
                            if total:
                                progress = f"{downloaded / total:.1%} ({progress}/{total / (1024 * 1024):.1f} MiB)"
                            logger.info(f"下载进度: {filename} {progress}, 平均 {speed:.1f} MiB/s")
                            last_progress = now
            elif media_check_ignore or mediatype[0] == "image":
                img = await response.aread()
                downloaded = len(img)
                if not is_thumbnail and compression and mediatype[1] in ["jpeg", "png"]:
                    pieces = split_long_image_bytes(img)
                    if pieces:
                        media.unlink(missing_ok=True)
                        piece_paths: list[Path] = []
                        for i, piece in enumerate(pieces):
                            piece_path = media.with_name(f"{media.stem}_p{i + 1}.jpg")
                            piece_path.write_bytes(piece)
                            piece_paths.append(piece_path)
                        logger.info(f"完成下载(切片): {media.name} -> {len(piece_paths)} 片")
                        return piece_paths
                if compression and mediatype[1] in ["jpeg", "png"]:
                    logger.info(f"压缩: {url} {mediatype[1]}")
                    if is_thumbnail:
                        img = compress(BytesIO(img), size=320, format="JPEG").getvalue()
                    else:
                        # 保持内容与 .jpg 文件名一致，并修正极端比例，避免本地 Bot API
                        # 把图片误判成 document。
                        img = compress(BytesIO(img), fix_ratio=True, format="JPEG").getvalue()
                with temp_media.open("wb") as file:
                    file.write(img)
            else:
                raise ValueError(f"媒体文件类型错误: {mediatype} {url}->{referer}")
            media.unlink(missing_ok=True)
            temp_media.rename(media)
            elapsed = max(time.monotonic() - download_started, 0.001)
            logger.info(
                f"完成下载: {media} ({downloaded / (1024 * 1024):.1f} MiB, "
                f"平均 {downloaded / elapsed / (1024 * 1024):.1f} MiB/s)"
            )
            return media
    except (asyncio.TimeoutError, httpx.TimeoutException) as err:
        logger.error(f"下载超时: {_display_media_url(url)}->{referer}")
        if raise_on_error:
            if isinstance(err, httpx.TimeoutException):
                raise
            raise httpx.TimeoutException(f"下载超时: {url}") from err
    except Exception as e:
        logger.error(f"下载错误: {_display_media_url(url)}->{referer}")
        logger.exception(e)
        if raise_on_error:
            raise
    finally:
        temp_media.unlink(missing_ok=True)


async def _probe_media_url(
    client: httpx.AsyncClient,
    referer: str,
    url: str,
    probe_bytes: int,
    probe_timeout: float,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[str, float] | None:
    header = BILIBILI_DESKTOP_HEADER.copy()
    header["Referer"] = referer
    header["Range"] = f"bytes=0-{probe_bytes - 1}"
    header["Accept-Encoding"] = "identity"
    downloaded = 0
    started = time.monotonic()

    async def probe() -> None:
        nonlocal downloaded
        async with timeout(probe_timeout), client.stream("GET", url, headers=header) as response:
            if response.status_code not in (200, 206):
                return
            async for chunk in response.aiter_raw():
                downloaded += len(chunk)
                if downloaded >= probe_bytes:
                    break

    try:
        if semaphore:
            async with semaphore:
                await probe()
        else:
            await probe()
    except asyncio.TimeoutError:
        pass
    except httpx.HTTPError:
        return None
    except Exception as err:
        logger.debug(f"UPOS测速失败: {_media_url_host(url)} ({type(err).__name__})")
        return None
    elapsed = max(time.monotonic() - started, 0.001)
    return (url, downloaded / elapsed) if downloaded else None


async def rank_media_urls(
    client: httpx.AsyncClient,
    referer: str,
    urls: list[str],
    *,
    probe_bytes: int = UPOS_PROBE_BYTES,
    probe_timeout: float = UPOS_PROBE_TIMEOUT,
    semaphore: asyncio.Semaphore | None = None,
) -> list[str]:
    """并发小范围测速，成功候选按实际吞吐排序，探测失败候选保留在末尾供兜底。"""
    candidates = _dedupe_media_urls(urls)
    if len(candidates) < 2 or probe_bytes <= 0 or probe_timeout <= 0:
        return candidates

    results = await asyncio.gather(
        *(_probe_media_url(client, referer, url, probe_bytes, probe_timeout, semaphore) for url in candidates)
    )
    speeds = {result[0]: result[1] for result in results if result is not None}
    if not speeds:
        logger.warning(f"UPOS测速均失败，按原顺序尝试: {', '.join(_media_url_host(url) for url in candidates)}")
        return candidates

    ranked = sorted(speeds, key=lambda url: speeds[url], reverse=True)
    ranked.extend(url for url in candidates if url not in speeds)
    summary = ", ".join(f"{_media_url_host(url)}={speeds[url] / (1024 * 1024):.1f} MiB/s" for url in speeds)
    logger.info(f"UPOS并发测速: {summary}; 选择 {_media_url_host(ranked[0])}")
    return ranked


def _parse_content_range(value: str) -> tuple[int, int, int] | None:
    match = _CONTENT_RANGE_RE.fullmatch(value.strip())
    return tuple(int(part) for part in match.groups()) if match else None


async def _inspect_range_source(
    client: httpx.AsyncClient,
    referer: str,
    url: str,
    semaphore: asyncio.Semaphore | None = None,
) -> _RangeSource | None:
    header = BILIBILI_DESKTOP_HEADER.copy()
    header["Referer"] = referer
    header["Range"] = "bytes=0-0"
    header["Accept-Encoding"] = "identity"

    async def inspect() -> _RangeSource | None:
        async with timeout(UPOS_PROBE_TIMEOUT), client.stream("GET", url, headers=header) as response:
            if response.status_code != 206:
                return None
            parsed = _parse_content_range(response.headers.get("content-range", ""))
            if not parsed or parsed[:2] != (0, 0) or parsed[2] <= 0:
                return None
            if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
                return None
            if response.headers.get("content-length") not in (None, "1"):
                return None
            body = bytearray()
            async for chunk in response.aiter_raw():
                body.extend(chunk)
                if len(body) > 1:
                    return None
            if len(body) != 1:
                return None
            return _RangeSource(
                url=url,
                total=parsed[2],
                content_type=response.headers.get("content-type", "").split(";", 1)[0].lower(),
                etag=response.headers.get("etag", ""),
            )

    try:
        if semaphore:
            async with semaphore:
                return await inspect()
        return await inspect()
    except (asyncio.TimeoutError, httpx.HTTPError):
        return None
    except Exception as err:
        logger.debug(f"Range能力探测失败: {_media_url_host(url)} ({type(err).__name__})")
        return None


async def _range_sources(
    client: httpx.AsyncClient,
    referer: str,
    urls: list[str],
    semaphore: asyncio.Semaphore | None = None,
) -> list[_RangeSource]:
    inspected = await asyncio.gather(*(_inspect_range_source(client, referer, url, semaphore) for url in urls))
    return [source for source in inspected if source and source.strong_etag]


def _build_range_chunks(total: int, workers: int) -> list[tuple[int, int, int]]:
    max_chunks = max(1, FILE_RANGE_MAX_CHUNKS)
    max_chunk_size = max(1, FILE_RANGE_MAX_CHUNK_SIZE)
    if total > max_chunks * max_chunk_size:
        return []
    target_chunks = min(max_chunks, max(workers, workers * max(1, FILE_RANGE_CHUNKS_PER_WORKER)))
    configured_chunk_size = min(max(1, FILE_RANGE_CHUNK_SIZE), max_chunk_size)
    chunk_size = min(configured_chunk_size, max(1, math.ceil(total / target_chunks)))
    chunk_size = max(chunk_size, math.ceil(total / max_chunks))
    return [
        (index, start, min(start + chunk_size - 1, total - 1))
        for index, start in enumerate(range(0, total, chunk_size))
    ]


async def _fetch_range(
    client: httpx.AsyncClient,
    referer: str,
    source: _RangeSource,
    start: int,
    end: int,
) -> bytes:
    header = BILIBILI_DESKTOP_HEADER.copy()
    header["Referer"] = referer
    header["Range"] = f"bytes={start}-{end}"
    header["Accept-Encoding"] = "identity"
    if source.strong_etag:
        header["If-Range"] = source.strong_etag

    try:
        async with timeout(FILE_RANGE_REQUEST_TIMEOUT), client.stream("GET", source.url, headers=header) as response:
            if response.status_code != 206:
                raise _RangeTransferError(f"Range响应状态错误: {_media_url_host(source.url)} {response.status_code}")
            parsed = _parse_content_range(response.headers.get("content-range", ""))
            if parsed != (start, end, source.total):
                raise _RangeTransferError(f"Content-Range不匹配: {_media_url_host(source.url)}")
            if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
                raise _RangeTransferError(f"Range响应被压缩: {_media_url_host(source.url)}")
            response_etag = response.headers.get("etag", "")
            if source.strong_etag and response_etag and response_etag != source.strong_etag:
                raise _RangeTransferError(f"Range资源版本变化: {_media_url_host(source.url)}")

            expected = end - start + 1
            content_length = response.headers.get("content-length")
            if content_length is not None and int(content_length) != expected:
                raise _RangeTransferError(f"Range长度头不匹配: {_media_url_host(source.url)}")
            body = bytearray()
            async for chunk in response.aiter_raw():
                body.extend(chunk)
                if len(body) > expected:
                    raise _RangeTransferError(f"Range响应超长: {_media_url_host(source.url)}")
            if len(body) != expected:
                raise _RangeTransferError(f"Range响应截断: {_media_url_host(source.url)}")
            return bytes(body)
    except _RangeTransferError:
        raise
    except (asyncio.TimeoutError, httpx.HTTPError, ValueError) as err:
        raise _RangeTransferError(
            f"Range请求失败: {_media_url_host(source.url)} ({type(err).__name__}: {err})"
        ) from err


async def _download_range_chunk(
    client: httpx.AsyncClient,
    referer: str,
    sources: list[_RangeSource],
    global_semaphore: asyncio.Semaphore,
    source_semaphore: asyncio.Semaphore,
    index: int,
    start: int,
    end: int,
) -> bytes:
    max_attempts = max(2, len(sources), FILE_RANGE_RETRIES + 1)
    source_order = [sources[(index + attempt) % len(sources)] for attempt in range(max_attempts)]
    active: dict[asyncio.Task, asyncio.Event] = {}
    next_attempt = 0
    last_error: Exception | None = None

    async def launch(source: _RangeSource, started: asyncio.Event) -> bytes:
        async with source_semaphore, global_semaphore:
            started.set()
            return await _fetch_range(client, referer, source, start, end)

    def start_attempt() -> None:
        nonlocal next_attempt
        event = asyncio.Event()
        task = asyncio.create_task(launch(source_order[next_attempt], event))
        active[task] = event
        next_attempt += 1

    start_attempt()
    try:
        while active:
            if len(active) == 1:
                only_event = next(iter(active.values()))
                if not only_event.is_set():
                    await only_event.wait()
            hedge_timeout = (
                FILE_RANGE_HEDGE_DELAY
                if len(active) == 1 and next_attempt < max_attempts and FILE_RANGE_HEDGE_DELAY > 0
                else None
            )
            done, pending = await asyncio.wait(
                active.keys(),
                timeout=hedge_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                start_attempt()
                continue

            active = {task: active[task] for task in pending}
            failed = False
            successful: bytes | None = None
            for task in done:
                try:
                    result = task.result()
                except Exception as err:
                    last_error = err
                    failed = True
                else:
                    successful = result

            if successful is not None:
                for pending_task in active:
                    pending_task.cancel()
                await asyncio.gather(*active.keys(), return_exceptions=True)
                return successful

            if failed and next_attempt < max_attempts and len(active) < 2:
                start_attempt()
        if last_error:
            raise last_error
        raise _RangeTransferError(f"Range分块下载失败: {start}-{end}")
    finally:
        for task in active:
            task.cancel()
        await asyncio.gather(*active.keys(), return_exceptions=True)


async def _download_ranges_from_source(
    client: httpx.AsyncClient,
    referer: str,
    source: _RangeSource,
    filename: str,
    semaphore: asyncio.Semaphore,
    workers: int,
) -> Path:
    total = source.total
    LOCAL_MEDIA_FILE_PATH.mkdir(parents=True, exist_ok=True)
    media = LOCAL_MEDIA_FILE_PATH / filename
    temp_media = LOCAL_MEDIA_FILE_PATH / uuid4().hex
    chunks = _build_range_chunks(total, workers)
    if not chunks:
        raise _RangeTransferError(f"Range分块限制无法容纳文件: {filename}")
    worker_count = min(workers, len(chunks))
    source_semaphore = asyncio.Semaphore(worker_count)
    write_lock = asyncio.Lock()
    completed = 0
    started = time.monotonic()
    last_progress = started
    tasks: list[asyncio.Task] = []
    logger.info(
        f"并发分块下载开始: {filename} {total / (1024 * 1024):.1f} MiB, "
        f"{len(chunks)} 块/{worker_count} 连接, UPOS: {_media_url_host(source.url)}"
    )

    try:
        with temp_media.open("w+b") as output:
            output.truncate(total)

            async def download_and_write(index: int, start: int, end: int) -> None:
                nonlocal completed, last_progress
                data = await _download_range_chunk(
                    client,
                    referer,
                    [source],
                    semaphore,
                    source_semaphore,
                    index,
                    start,
                    end,
                )
                async with write_lock:
                    output.seek(start)
                    output.write(data)
                    completed += len(data)
                    now = time.monotonic()
                    if DOWNLOAD_PROGRESS_INTERVAL > 0 and now - last_progress >= DOWNLOAD_PROGRESS_INTERVAL:
                        speed = completed / max(now - started, 0.001) / (1024 * 1024)
                        logger.info(
                            f"分块下载进度: {filename} {completed / total:.1%} "
                            f"({completed / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MiB), "
                            f"平均 {speed:.1f} MiB/s"
                        )
                        last_progress = now

            chunk_queue = asyncio.Queue()
            for chunk in chunks:
                chunk_queue.put_nowait(chunk)

            async def download_worker() -> None:
                while True:
                    try:
                        chunk = chunk_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await download_and_write(*chunk)
                    finally:
                        chunk_queue.task_done()

            tasks = [asyncio.create_task(download_worker()) for _ in range(worker_count)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            output.flush()

        if completed != total or temp_media.stat().st_size != total:
            raise _RangeTransferError(f"分块下载文件不完整: {filename}")
        temp_media.replace(media)
        elapsed = max(time.monotonic() - started, 0.001)
        logger.info(
            f"完成并发分块下载: {media} ({total / (1024 * 1024):.1f} MiB, "
            f"平均 {total / elapsed / (1024 * 1024):.1f} MiB/s)"
        )
        return media
    finally:
        temp_media.unlink(missing_ok=True)


async def get_media_by_ranges(
    client: httpx.AsyncClient,
    referer: str,
    urls: list[str],
    filename: str,
    *,
    range_semaphore: asyncio.Semaphore | None = None,
) -> Path | None:
    workers = _range_worker_count()
    if workers <= 1 or FILE_RANGE_CHUNK_SIZE <= 0:
        return None

    semaphore = range_semaphore or asyncio.Semaphore(workers)
    sources = await _range_sources(client, referer, urls, semaphore)
    eligible = [
        source
        for source in sources
        if max(1, FILE_RANGE_MIN_SIZE) <= source.total <= max(1, FILE_RANGE_MAX_SIZE)
        and source.content_type.split("/", 1)[0] in ("video", "audio", "application")
    ]
    if not eligible:
        return None

    last_error: _RangeTransferError | None = None
    worker_levels = list(dict.fromkeys((workers, max(1, workers // 2), 1)))
    for source in eligible:
        for index, attempt_workers in enumerate(worker_levels):
            try:
                return await _download_ranges_from_source(
                    client,
                    referer,
                    source,
                    filename,
                    semaphore,
                    attempt_workers,
                )
            except _RangeTransferError as err:
                last_error = err
                if index + 1 < len(worker_levels):
                    logger.warning(
                        f"分块下载失败，降低并发到 {worker_levels[index + 1]} 后重试: "
                        f"{_media_url_host(source.url)} ({err})"
                    )
                else:
                    logger.warning(f"分块源失败，尝试下一 UPOS: {_media_url_host(source.url)} ({err})")
    if last_error:
        raise last_error
    return None


async def get_media_from_candidates(
    client: httpx.AsyncClient,
    referer: str,
    urls: list[str],
    filename: str,
    compression: bool = True,
    media_check_ignore: bool = False,
    no_cache: bool = False,
    is_thumbnail: bool = False,
    cache_lookup: CacheLookup | None = None,
    cache_key: str | None = None,
    raise_on_error: bool = False,
    range_semaphore: asyncio.Semaphore | None = None,
    parallel_ranges: bool = False,
) -> Path | str | list[Path] | None:
    """测速选择最快候选；完整下载失败时自动从下一个 UPOS 重新下载。"""
    if not no_cache and cache_lookup is not None:
        file_id = await cache_lookup(cache_key or filename)
        if file_id:
            return file_id

    if parallel_ranges and range_semaphore is None:
        range_semaphore = asyncio.Semaphore(_range_worker_count())
    ranked = await rank_media_urls(
        client,
        referer,
        urls,
        semaphore=range_semaphore if parallel_ranges else None,
    )
    if parallel_ranges:
        try:
            ranged = await get_media_by_ranges(
                client,
                referer,
                ranked,
                filename,
                range_semaphore=range_semaphore,
            )
            if ranged:
                return ranged
        except asyncio.CancelledError:
            raise
        except _RangeTransferError as err:
            logger.warning(f"并发分块下载失败，退回单流: {filename} ({err})")

    last_error: Exception | None = None
    for index, url in enumerate(ranked):
        try:
            result = await get_media(
                client,
                referer,
                url,
                filename,
                compression=compression,
                media_check_ignore=media_check_ignore,
                no_cache=True,
                is_thumbnail=is_thumbnail,
                raise_on_error=True,
            )
            if result:
                return result
        except Exception as err:
            last_error = err
            next_host = _media_url_host(ranked[index + 1]) if index + 1 < len(ranked) else None
            suffix = f"，切换到 {next_host}" if next_host else ""
            logger.warning(f"UPOS下载失败: {_media_url_host(url)} ({type(err).__name__}){suffix}")

    if raise_on_error:
        if last_error:
            raise last_error
        raise httpx.RequestError(f"所有 UPOS 候选均下载失败: {referer}")
    return None


def split_long_image_bytes(
    raw: bytes,
    ratio: float = IMAGE_SPLIT_RATIO,
    max_pieces: int = IMAGE_SPLIT_MAX_PIECES,
) -> list[bytes] | None:
    """在压缩前将超长图切成有限数量的全宽 JPEG；普通图片返回 ``None``。"""
    try:
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or height / width <= ratio:
                return None

            max_pieces = max(1, max_pieces)
            slice_height = int(width * ratio)
            count = math.ceil(height / slice_height)
            if count > max_pieces:
                count = max_pieces
                slice_height = math.ceil(height / count)

            image = image.convert("RGB")
            pieces: list[bytes] = []
            for index in range(count):
                top = index * slice_height
                bottom = min(top + slice_height, height)
                if top >= bottom:
                    break
                crop = image.crop((0, top, width, bottom))
                if crop.size[0] + crop.size[1] > 10000:
                    crop.thumbnail((4000, 8000), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                crop.save(buffer, "JPEG", optimize=True)
                pieces.append(buffer.getvalue())
            logger.info(f"超长图切割: {width}x{height} -> {len(pieces)} 片")
            return pieces or None
    except Exception as err:
        logger.error(f"超长图切割失败，按原图压缩: {err}")
        return None


def expand_long_images(content: ParsedContent, media: list) -> list:
    """展开 ``get_media`` 返回的切片列表，并保持 urls/filenames 与媒体索引对齐。"""
    if not content.media or not media:
        return media

    new_media: list = []
    new_urls: list = []
    new_filenames: list = []
    new_candidates: list[list[str]] = []
    for index, item in enumerate(media):
        url = content.media.urls[index] if index < len(content.media.urls) else ""
        filename = content.media.filenames[index] if index < len(content.media.filenames) else ""
        candidates = (
            content.media.url_candidates[index] if index < len(content.media.url_candidates) else ([url] if url else [])
        )
        if isinstance(item, list):
            for piece_index, piece in enumerate(item):
                new_media.append(piece)
                new_urls.append(url)
                new_candidates.append(candidates)
                new_filenames.append(
                    f"{Path(filename).stem or 'image'}_p{piece_index + 1}.jpg"
                    if filename
                    else getattr(piece, "name", filename)
                )
        else:
            new_media.append(item)
            new_urls.append(url)
            new_candidates.append(candidates)
            new_filenames.append(filename)
    content.media.urls = new_urls
    content.media.filenames = new_filenames
    content.media.url_candidates = new_candidates
    return new_media


def _merged_media_filename(content: ParsedContent) -> str:
    base_name = content.media.filenames[0] if content.media and content.media.filenames else "merged"
    return f"{Path(base_name).stem}_merged.mp4"


def _content_media_candidates(content: ParsedContent, index: int, primary: str) -> list[str]:
    candidates = (
        content.media.url_candidates[index] if content.media and index < len(content.media.url_candidates) else []
    )
    return _dedupe_media_urls([primary, *candidates])


async def handle_dash_media(
    f: ParsedContent,
    client: httpx.AsyncClient,
    cache_lookup: CacheLookup | None = None,
    cache_key_builder: CacheKeyBuilder | None = None,
    document: bool = False,
    range_semaphore: asyncio.Semaphore | None = None,
):
    """处理 DASH 视频合并（多轨流下载后用 ffmpeg 合并）"""
    if not f.media or not f.media.merge_streams:
        return []
    if len(f.media.urls) < 2:
        raise httpx.RequestError(f"DASH媒体轨道不足: {f.url}")
    range_semaphore = range_semaphore or asyncio.Semaphore(_range_worker_count())
    res = []
    download_tasks: list[asyncio.Task] = []
    try:
        # Use a distinct merged filename to avoid ffmpeg reading and writing the same file
        merged_name = _merged_media_filename(f)
        cache_dash_file = LOCAL_MEDIA_FILE_PATH / merged_name

        if cache_lookup is not None:
            cache_key = (
                cache_key_builder(cache_dash_file.name, f.media.type, None, document)
                if cache_key_builder
                else cache_dash_file.name
            )
            cache_dash = await cache_lookup(cache_key)
            if cache_dash:
                f.media.urls = [str(cache_dash_file.absolute())]
                f.media.filenames = [cache_dash_file.name]
                f.media.url_candidates = []
                f.media.merge_streams = False
                return [cache_dash]

        download_tasks = [
            asyncio.create_task(
                get_media_from_candidates(
                    client,
                    f.url,
                    _content_media_candidates(f, index, media_url),
                    filename,
                    no_cache=True,
                    cache_lookup=cache_lookup,
                    raise_on_error=True,
                    range_semaphore=range_semaphore,
                    parallel_ranges=True,
                )
            )
            for index, (media_url, filename) in enumerate(zip(f.media.urls, f.media.filenames, strict=False))
        ]
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        res = [result for result in results if result and not isinstance(result, BaseException)]
        if failures:
            failure = failures[0]
            if isinstance(failure, httpx.HTTPError):
                raise failure
            raise httpx.RequestError(f"DASH媒体下载失败: {f.url}") from failure
        if len(res) < 2:
            raise httpx.RequestError(f"DASH媒体下载不完整: {f.url}")
        cmd = [os.environ.get("FFMPEG_PATH", "ffmpeg"), "-y"]
        for item in res:
            cmd.extend(["-i", str(item)])
        cmd.extend(["-vcodec", "copy", "-acodec", "copy", str(cache_dash_file.absolute())])
        logger.info(f"开始合并，执行命令：{' '.join(cmd)}")
        subprocess.run(cmd, check=True)  # noqa: ASYNC221, S603

        f.media.urls = [str(cache_dash_file.absolute())]
        f.media.filenames = [cache_dash_file.name]
        f.media.url_candidates = []
        f.media.merge_streams = False
        logger.debug(f"合并完成: {f.url}")
        return [cache_dash_file]
    except subprocess.CalledProcessError as e:
        logger.error(f"DASH媒体处理失败: {f.url} - {e!s}")
        raise httpx.RequestError(f"DASH媒体合并失败: {f.url}") from e
    finally:
        for task in download_tasks:
            if not task.done():
                task.cancel()
        completed = await asyncio.gather(*download_tasks, return_exceptions=True)
        for item in completed:
            if isinstance(item, Path):
                item.unlink(missing_ok=True)


async def get_media_for_content(
    f: ParsedContent,
    compression: bool = True,
    media_check_ignore: bool = False,
    no_media: bool = False,
    cache_lookup: CacheLookup | None = None,
    cache_key_builder: CacheKeyBuilder | None = None,
) -> tuple[list, Path | str | None]:
    """下载并准备媒体文件，返回 (media_list, thumbnail)"""
    if not f.media:
        return [], None
    if not f.media.urls:
        return [], None

    async with httpx.AsyncClient(
        # 大文件 Range 并发需要独立 HTTP/1.1 连接；HTTP/2 会把多个块复用到
        # 同一 TCP 连接，部分 Akamai UPOS 一次 stream reset 就会让所有块同时失败。
        http2=False,
        follow_redirects=True,
        proxy=os.environ.get("FILE_PROXY", os.environ.get("HTTP_PROXY")),
        timeout=httpx.Timeout(
            connect=FILE_CONNECT_TIMEOUT,
            read=FILE_READ_TIMEOUT,
            write=FILE_WRITE_TIMEOUT,
            pool=FILE_POOL_TIMEOUT,
        ),
    ) as client:
        range_semaphore = asyncio.Semaphore(_range_worker_count())
        mediathumb = None
        if f.media.thumbnail:
            if f.media.need_download or LOCAL_MODE:
                mediathumb = await get_media(
                    client,
                    f.url,
                    f.media.thumbnail,
                    f.media.thumbnail_filename,
                    compression=compression,
                    media_check_ignore=False,
                    no_cache=True,
                    is_thumbnail=True,
                    cache_lookup=cache_lookup,
                    raise_on_error=media_check_ignore,
                )
            else:
                mediathumb = referer_url(f.media.thumbnail, f.url)

        media = []
        if no_media:
            return media, mediathumb

        if f.media.merge_streams:
            # DASH 多轨流必须下载后合并，无论是否 local 模式
            try:
                media = await handle_dash_media(
                    f,
                    client,
                    cache_lookup=cache_lookup,
                    cache_key_builder=cache_key_builder,
                    document=media_check_ignore,
                    range_semaphore=range_semaphore,
                )
            except httpx.HTTPError as err:
                if not f.media.fallback_url:
                    raise
                logger.warning(f"DASH媒体下载失败，改用 MP4 直链: {f.url} - {err}")
                fallback_url = f.media.fallback_url
                fallback_filename = _merged_media_filename(f)
                fallback = await get_media_from_candidates(
                    client,
                    f.url,
                    _dedupe_media_urls([fallback_url, *f.media.fallback_candidates]),
                    fallback_filename,
                    compression=compression,
                    media_check_ignore=media_check_ignore,
                    no_cache=True,
                    cache_lookup=cache_lookup,
                    raise_on_error=True,
                    range_semaphore=range_semaphore,
                    parallel_ranges=True,
                )
                if not fallback or isinstance(fallback, list):
                    raise httpx.RequestError(f"MP4 回退媒体下载失败: {f.url}") from err
                f.media.urls = [fallback_url]
                f.media.filenames = [fallback_filename]
                f.media.url_candidates = []
                f.media.fallback_candidates = []
                f.media.merge_streams = False
                media = [fallback]
            if media:
                return media, mediathumb
        elif f.media.need_download or LOCAL_MODE:
            # video/audio 以及 /fetch 原文件都必须完整下载；失败需交给上传队列重试，
            # 不能静默过滤成空列表后继续调用平台上传接口。
            required_media = media_check_ignore or f.media.type in ["video", "audio"]
            tasks = [
                get_media_from_candidates(
                    client,
                    f.url,
                    _content_media_candidates(f, index, media_url),
                    fn,
                    compression=compression,
                    media_check_ignore=media_check_ignore,
                    cache_lookup=cache_lookup,
                    cache_key=(
                        cache_key_builder(fn, f.media.type, media_url, media_check_ignore) if cache_key_builder else fn
                    ),
                    raise_on_error=required_media,
                    range_semaphore=range_semaphore,
                    parallel_ranges=f.media.type in ["video", "audio"],
                )
                for index, (media_url, fn) in enumerate(zip(f.media.urls, f.media.filenames, strict=False))
            ]
            downloaded = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [item for item in downloaded if isinstance(item, Exception)]
            kept_urls: list = []
            kept_filenames: list = []
            kept_candidates: list[list[str]] = []
            media = []
            for index, (item, media_url, filename) in enumerate(
                zip(downloaded, f.media.urls, f.media.filenames, strict=False)
            ):
                if item and not isinstance(item, BaseException):
                    media.append(item)
                    kept_urls.append(media_url)
                    kept_filenames.append(filename)
                    kept_candidates.append(_content_media_candidates(f, index, media_url))
            if required_media and (failures or not media):
                cleanup_medias(media)
                if failures:
                    raise failures[0]
                raise httpx.RequestError(f"媒体下载失败: {f.url}")
            f.media.urls = kept_urls
            f.media.filenames = kept_filenames
            f.media.url_candidates = kept_candidates
            media = expand_long_images(f, media)
        else:
            if f.media.type in ["video", "audio"]:
                media = [referer_url(f.media.urls[0], f.url)]
            else:
                media = f.media.urls

        return media, mediathumb
