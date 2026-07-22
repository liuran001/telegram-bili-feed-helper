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
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine
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

CacheLookup = Callable[[str], Coroutine[None, None, str | None]]
CacheKeyBuilder = Callable[[str, str | None, Path | str | None, bool], str]


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
) -> tuple[str, float] | None:
    header = BILIBILI_DESKTOP_HEADER.copy()
    header["Referer"] = referer
    header["Range"] = f"bytes=0-{probe_bytes - 1}"
    header["Accept-Encoding"] = "identity"
    downloaded = 0
    started = time.monotonic()
    try:
        async with timeout(probe_timeout), client.stream("GET", url, headers=header) as response:
            if response.status_code not in (200, 206):
                return None
            async for chunk in response.aiter_raw():
                downloaded += len(chunk)
                if downloaded >= probe_bytes:
                    break
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
) -> list[str]:
    """并发小范围测速，成功候选按实际吞吐排序，探测失败候选保留在末尾供兜底。"""
    candidates = _dedupe_media_urls(urls)
    if len(candidates) < 2 or probe_bytes <= 0 or probe_timeout <= 0:
        return candidates

    results = await asyncio.gather(
        *(_probe_media_url(client, referer, url, probe_bytes, probe_timeout) for url in candidates)
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
) -> Path | str | list[Path] | None:
    """测速选择最快候选；完整下载失败时自动从下一个 UPOS 重新下载。"""
    if not no_cache and cache_lookup is not None:
        file_id = await cache_lookup(cache_key or filename)
        if file_id:
            return file_id

    ranked = await rank_media_urls(client, referer, urls)
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
):
    """处理 DASH 视频合并（多轨流下载后用 ffmpeg 合并）"""
    if not f.media or not f.media.merge_streams:
        return []
    if len(f.media.urls) < 2:
        raise httpx.RequestError(f"DASH媒体轨道不足: {f.url}")
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
        http2=True,
        follow_redirects=True,
        proxy=os.environ.get("FILE_PROXY", os.environ.get("HTTP_PROXY")),
        timeout=httpx.Timeout(
            connect=FILE_CONNECT_TIMEOUT,
            read=FILE_READ_TIMEOUT,
            write=FILE_WRITE_TIMEOUT,
            pool=FILE_POOL_TIMEOUT,
        ),
    ) as client:
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
