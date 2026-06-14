import asyncio
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from async_timeout import timeout
from PIL import Image
from telegram import InputMediaDocument, InputMediaPhoto, InputMediaVideo, Message
from telegram.constants import ChatType
from telegram.error import BadRequest, NetworkError, RetryAfter
from tqdm import tqdm

from ...model import MediaConstraints, ParsedContent
from ...provider.bilibili.api import BILIBILI_DESKTOP_HEADER, CACHES_TIMER, referer_url
from ...storage.cache import RedisCache
from ...storage.models import TelegramFileCache
from ...utils import compress, logger

LOCAL_MEDIA_FILE_PATH = Path(os.environ.get("LOCAL_TEMP_FILE_PATH", str(Path.cwd()))) / ".tmp"
LOCAL_MODE = bool(os.environ.get("LOCAL_MODE", False))

# 超长图切割：高宽比超过该值则按高度切成多片，使每片接近此比例（默认 2，保证切片清晰可读）。
# Telegram photo 限制 width+height<=10000，且过细长条会被本地 Bot API 判为 document。
IMAGE_SPLIT_RATIO = float(os.environ.get("IMAGE_SPLIT_RATIO", 2))
# 单条动态每张超长图切割后最多片数（Telegram media group 单组上限 10），防止极端长图刷屏
IMAGE_SPLIT_MAX_PIECES = int(os.environ.get("IMAGE_SPLIT_MAX_PIECES", 10))

BILIBILI_SHARE_URL_REGEX = r"(?i)【.*】 https://[\w\.]*?(?:bilibili\.com|b23\.tv|bili2?2?3?3?\.cn)\S+"


def _get_constraints() -> MediaConstraints:
    return MediaConstraints(
        max_upload_size=2 * 1024 * 1024 * 1024 if LOCAL_MODE else 50 * 1024 * 1024,
        max_download_size=2 * 1024 * 1024 * 1024,
        caption_max_length=1024,
        local_mode=LOCAL_MODE,
    )


@dataclass
class UploadTask:
    """上传任务数据结构"""

    user_id: int
    message: Message
    parsed_content: ParsedContent
    media: list[Path | str]
    mediathumb: Path | str | None
    is_parse_cmd: bool
    is_video_cmd: bool
    urls: list[str]
    task_type: str = "parse"
    fetch_mode: str | None = None
    task_id: str = field(default_factory=lambda: uuid4().hex)
    cancelled: bool = field(default=False)


# Telegram file_id 与媒体类型强绑定：同一个 file_id 不能跨类型复用（本地 Bot API 会在构造
# InputMediaPhoto 时直接拒绝一个 document 类型的 file_id，报 "Can't use file of type document as photo"）。
# parse 路径按 photo/video/audio/gif 发送，/fetch 路径按 document 发送，二者对同一图片 URL 派生出
# 相同的 filename，若共用缓存 key 就会串号。这里给缓存主键加类型前缀做命名空间隔离。
# mediafilename 主键为 CharField(64)，真实文件名最长约 44 字符，加前缀后仍在上限内；极端情况下截断保护。
_CACHE_KEY_MAXLEN = 64


def _cache_key(filename: str, kind: str) -> str:
    return f"{kind}:{filename}"[:_CACHE_KEY_MAXLEN]


def _send_kind(media_type: str | None, url: Path | str | None = None, document: bool = False) -> str:
    """推断一个媒体最终以哪种 Telegram 类型发送，作为缓存命名空间。
    /fetch 路径统一发 document；parse 路径按 media.type 与 .gif 后缀区分 video/audio/gif/photo。"""
    if document:
        return "document"
    if media_type == "video":
        return "video"
    if media_type == "audio":
        return "audio"
    if url is not None and ".gif" in str(url):
        return "gif"
    return "photo"


async def get_cached_media_file_id(filename: str, kind: str = "photo") -> str | None:
    file = await TelegramFileCache.get_or_none(mediafilename=_cache_key(filename, kind))
    if file:
        return file.file_id
    return None


async def cache_media(mediafilename: str, file, kind: str = "photo") -> None:
    if not file:
        return
    try:
        await TelegramFileCache.update_or_create(
            mediafilename=_cache_key(mediafilename, kind), defaults=dict(file_id=file.file_id)
        )
    except Exception as e:
        logger.exception(e)


def cleanup_medias(medias):
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
    cache_kind: str = "photo",
) -> Path | str | None:
    if isinstance(url, Path):
        return url
    if not no_cache:
        file_id = await get_cached_media_file_id(filename, cache_kind)
        if file_id:
            return file_id
    LOCAL_MEDIA_FILE_PATH.mkdir(parents=True, exist_ok=True)
    media = LOCAL_MEDIA_FILE_PATH / filename
    temp_media = LOCAL_MEDIA_FILE_PATH / uuid4().hex
    try:
        header = BILIBILI_DESKTOP_HEADER.copy()
        header["Referer"] = referer
        async with timeout(CACHES_TIMER["LOCK"]), client.stream("GET", url, headers=header) as response:
            logger.info(f"下载开始: {url}")
            if response.status_code != 200:
                raise NetworkError(f"媒体文件获取错误: {response.status_code} {url}->{referer}")
            content_type = response.headers.get("content-type")
            if content_type is None:
                raise NetworkError(f"媒体文件获取错误: 无法获取 content-type {url}->{referer}")
            mediatype = content_type.split("/")
            total = int(response.headers.get("content-length", 0))
            if mediatype[0] in ["video", "audio", "application"]:
                with (
                    temp_media.open("wb") as file,
                    tqdm(
                        total=total,
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                        desc=response.request.url.host + "->" + filename,
                    ) as pbar,
                ):
                    async for chunk in response.aiter_bytes():
                        file.write(chunk)
                        pbar.update(len(chunk))
            elif media_check_ignore or mediatype[0] == "image":
                img = await response.aread()
                # 非缩略图：先判断是否超长图，是则在压缩前对原图切片，逐片返回多个文件
                if not is_thumbnail and compression and mediatype[1] in ["jpeg", "png"]:
                    pieces = split_long_image_bytes(img)
                    if pieces:
                        piece_paths: list[Path] = []
                        for i, pbytes in enumerate(pieces):
                            pp = media.with_name(f"{media.stem}_p{i + 1}.jpg")
                            with pp.open("wb") as pf:
                                pf.write(pbytes)
                            piece_paths.append(pp)
                        logger.info(f"完成下载(切片): {media.name} -> {len(piece_paths)} 片")
                        return piece_paths
                if compression and mediatype[1] in ["jpeg", "png"]:
                    logger.info(f"压缩: {url} {mediatype[1]}")
                    if is_thumbnail:
                        img = compress(BytesIO(img), size=320, format="JPEG").getvalue()
                    else:
                        # 用 JPEG 保持与 .jpg 文件名一致（默认 PNG 会让本地 Bot API 按扩展名误判为 document）；
                        # fix_ratio=True 兜底修正极端比例图
                        img = compress(BytesIO(img), fix_ratio=True, format="JPEG").getvalue()
                with temp_media.open("wb") as file:
                    file.write(img)
            else:
                raise NetworkError(f"媒体文件类型错误: {mediatype} {url}->{referer}")
            media.unlink(missing_ok=True)
            temp_media.rename(media)
            logger.info(f"完成下载: {media}")
            return media
    except asyncio.TimeoutError:
        logger.error(f"下载超时: {url}->{referer}")
        raise NetworkError(f"下载超时: {url}")
    except Exception as e:
        logger.error(f"下载错误: {url}->{referer}")
        logger.exception(e)
    finally:
        temp_media.unlink(missing_ok=True)


async def handle_dash_media(f: ParsedContent, client: httpx.AsyncClient):
    """处理 DASH 视频合并（多轨流下载后用 ffmpeg 合并）"""
    if not f.media or not f.media.merge_streams or len(f.media.urls) < 2:
        return []
    res = []
    try:
        # Use a distinct merged filename to avoid ffmpeg reading and writing the same file
        base_name = f.media.filenames[0] if f.media.filenames else "merged"
        merged_name = Path(base_name).stem + "_merged.mp4"
        cache_dash_file = LOCAL_MEDIA_FILE_PATH / merged_name

        cache_dash = await get_cached_media_file_id(cache_dash_file.name, "video")
        if cache_dash:
            f.media.urls = [str(cache_dash_file.absolute())]
            f.media.filenames = [cache_dash_file.name]
            f.media.merge_streams = False
            return [cache_dash]

        tasks = [
            get_media(client, f.url, m, fn, no_cache=True)
            for m, fn in zip(f.media.urls, f.media.filenames, strict=False)
        ]
        res = [m for m in await asyncio.gather(*tasks) if m]
        if len(res) < 2:
            logger.error(f"DASH媒体下载失败: {f.url}")
            return []
        cmd = [os.environ.get("FFMPEG_PATH", "ffmpeg"), "-y"]
        for item in res:
            cmd.extend(["-i", str(item)])
        cmd.extend(["-vcodec", "copy", "-acodec", "copy", str(cache_dash_file.absolute())])
        logger.info(f"开始合并，执行命令：{' '.join(cmd)}")
        subprocess.run(cmd, check=True)  # noqa: ASYNC221, S603

        f.media.urls = [str(cache_dash_file.absolute())]
        f.media.filenames = [cache_dash_file.name]
        f.media.merge_streams = False
        logger.debug(f"合并完成: {f.url}")
        return [cache_dash_file]
    except subprocess.CalledProcessError as e:
        logger.error(f"DASH媒体处理失败: {f.url} - {e!s}")
        return []
    finally:
        for item in res:
            if isinstance(item, Path):
                item.unlink(missing_ok=True)


def split_long_image_bytes(
    raw: bytes, ratio: float = IMAGE_SPLIT_RATIO, max_pieces: int = IMAGE_SPLIT_MAX_PIECES
) -> list[bytes] | None:
    """对原始图片字节，若高宽比 > ratio 则按高度切成多片（在压缩之前对原图切，避免细条信息损毁），
    每片单独编码为 JPEG bytes 返回。非超长图返回 None（表示无需切割，走常规压缩）。"""
    try:
        with Image.open(BytesIO(raw)) as im:
            w, h = im.size
            if w <= 0 or h <= 0 or h / w <= ratio:
                return None
            slice_h = int(w * ratio)  # 每片目标高度，使切片比例接近 ratio
            n = math.ceil(h / slice_h)
            if n > max_pieces:
                # 超过上限：放大每片高度，保证总片数不超过 max_pieces（末片可能略超比例，可接受）
                n = max_pieces
                slice_h = math.ceil(h / n)
            im = im.convert("RGB")
            pieces: list[bytes] = []
            for i in range(n):
                top = i * slice_h
                bottom = min(top + slice_h, h)
                if top >= bottom:
                    break
                crop = im.crop((0, top, w, bottom))
                # 仅当切片仍超出 tdlib 约束(w+h<=10000)时才等比缩小（极端长图+片数上限才会触发）
                if crop.size[0] + crop.size[1] > 10000:
                    crop.thumbnail((4000, 8000), Image.Resampling.LANCZOS)
                buf = BytesIO()
                crop.save(buf, "JPEG", optimize=True)
                pieces.append(buf.getvalue())
            logger.info(f"超长图切割: {w}x{h} -> {len(pieces)} 片")
            return pieces or None
    except Exception as e:
        logger.error(f"超长图切割失败，按原图压缩: {e}")
        return None


def expand_long_images(f: ParsedContent, media: list) -> list:
    """把 media/urls/filenames 中已切片的占位项展开，保持三者索引对齐。
    切片产物由 get_media 以 list[Path] 形式返回，这里把嵌套 list 摊平并复制对应的 url。"""
    if not f.media or not media:
        return media
    new_media: list = []
    new_urls: list = []
    new_filenames: list = []
    urls = f.media.urls
    filenames = f.media.filenames
    for idx, item in enumerate(media):
        url = urls[idx] if idx < len(urls) else ""
        fn = filenames[idx] if idx < len(filenames) else ""
        if isinstance(item, list):
            # 一张超长图被切成多片
            for j, p in enumerate(item):
                new_media.append(p)
                new_urls.append(url)
                new_filenames.append(f"{Path(fn).stem or 'image'}_p{j + 1}.jpg" if fn else getattr(p, "name", fn))
        else:
            new_media.append(item)
            new_urls.append(url)
            new_filenames.append(fn)
    f.media.urls = new_urls
    f.media.filenames = new_filenames
    return new_media


async def get_media_for_content(
    f: ParsedContent,
    compression: bool = True,
    media_check_ignore: bool = False,
    no_media: bool = False,
) -> tuple[list, Path | str | None]:
    """下载并准备媒体文件（替代原 get_media_mediathumb_by_parser）"""
    if not f.media:
        return [], None

    async with httpx.AsyncClient(
        http2=True,
        follow_redirects=True,
        proxy=os.environ.get("FILE_PROXY", os.environ.get("HTTP_PROXY")),
    ) as client:
        # Handle thumbnail
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
                )
            else:
                mediathumb = referer_url(f.media.thumbnail, f.url)

        # Handle main media
        media = []
        if no_media:
            return media, mediathumb

        if f.media.merge_streams:
            # DASH 多轨流必须下载后合并，无论是否 local 模式
            media = await handle_dash_media(f, client)
            if media:
                return media, mediathumb
        elif f.media.need_download or LOCAL_MODE:
            tasks = [
                get_media(
                    client,
                    f.url,
                    m,
                    fn,
                    compression=compression,
                    media_check_ignore=media_check_ignore,
                    cache_kind=_send_kind(f.media.type, m, document=media_check_ignore),
                )
                for m, fn in zip(f.media.urls, f.media.filenames, strict=False)
            ]
            raw_media = await asyncio.gather(*tasks)
            # 过滤下载失败(None)的项时，同步剔除对应的 url/filename，保持索引对齐
            kept_urls: list = []
            kept_fns: list = []
            media = []
            for m, u, fn in zip(raw_media, f.media.urls, f.media.filenames, strict=False):
                if m:
                    media.append(m)
                    kept_urls.append(u)
                    kept_fns.append(fn)
            f.media.urls = kept_urls
            f.media.filenames = kept_fns
            # 下载后把超长图切成多张，避免被 Telegram 判为 document
            media = expand_long_images(f, media)
        else:
            if f.media.type in ["video", "audio"]:
                media = [referer_url(f.media.urls[0], f.url)]
            else:
                media = f.media.urls

        return media, mediathumb


class UploadQueueManager:
    """上传队列管理器 - 处理 Telegram API 限流"""

    def __init__(self, max_workers: int = 4, max_user_tasks: int = 5, max_queue_size: int = 200):
        self.queue: asyncio.Queue[UploadTask] = asyncio.Queue(maxsize=max_queue_size)
        self.max_workers = max_workers
        self.max_user_tasks = max_user_tasks
        self.active_tasks: dict[int, dict[str, UploadTask]] = {}
        self.processing_tasks: dict[int, dict[str, asyncio.Task]] = {}
        self.workers: list[asyncio.Task] = []
        self._lock = asyncio.Lock()

    async def submit(self, task: UploadTask) -> None:
        async with self._lock:
            if task.user_id not in self.active_tasks:
                self.active_tasks[task.user_id] = {}
            user_tasks = self.active_tasks[task.user_id]
            if len(user_tasks) >= self.max_user_tasks:
                logger.warning(f"用户 {task.user_id} 的任务数已达上限 ({self.max_user_tasks})，丢弃新任务")
                return
            for existing_task in user_tasks.values():
                if existing_task.urls == task.urls:
                    logger.info(f"用户 {task.user_id} 提交了重复的任务，忽略")
                    return
            self.active_tasks[task.user_id][task.task_id] = task
        await self.queue.put(task)
        logger.info(
            f"任务 {task.task_id[:8]} 已提交 (用户: {task.user_id}, 类型: {task.task_type}), 队列深度: {self.queue.qsize()}"
        )

    async def cancel_user_tasks(self, user_id: int) -> int:
        async with self._lock:
            cancelled_count = 0
            if user_id in self.processing_tasks:
                for task in self.processing_tasks[user_id].values():
                    task.cancel()
                    cancelled_count += 1
                del self.processing_tasks[user_id]
            if user_id in self.active_tasks:
                cancelled_count += len(self.active_tasks[user_id])
                del self.active_tasks[user_id]
            if cancelled_count > 0:
                logger.info(f"用户 {user_id} 手动取消了 ({cancelled_count} 个任务)")
            return cancelled_count

    async def get_user_tasks(self, user_id: int) -> list[str]:
        async with self._lock:
            tasks = self.active_tasks.get(user_id, {})
            return [f"{t.parsed_content.url} (ID: {t.task_id[:8]})" for t in tasks.values()]

    async def _worker(self, worker_id: int) -> None:
        logger.info(f"上传 Worker {worker_id} 启动")
        while True:
            try:
                task = await self.queue.get()
                async with self._lock:
                    user_tasks = self.active_tasks.get(task.user_id, {})
                    if task.task_id not in user_tasks:
                        self.queue.task_done()
                        continue
                process_task = asyncio.create_task(self._process_upload(task))
                async with self._lock:
                    if task.user_id not in self.processing_tasks:
                        self.processing_tasks[task.user_id] = {}
                    self.processing_tasks[task.user_id][task.task_id] = process_task
                try:
                    await process_task
                except asyncio.CancelledError:
                    if not process_task.cancelled():
                        raise
                finally:
                    async with self._lock:
                        if (
                            task.user_id in self.processing_tasks
                            and task.task_id in self.processing_tasks[task.user_id]
                        ):
                            del self.processing_tasks[task.user_id][task.task_id]
                            if not self.processing_tasks[task.user_id]:
                                del self.processing_tasks[task.user_id]
                async with self._lock:
                    if task.user_id in self.active_tasks and task.task_id in self.active_tasks[task.user_id]:
                        del self.active_tasks[task.user_id][task.task_id]
                        if not self.active_tasks[task.user_id]:
                            del self.active_tasks[task.user_id]
                self.queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} 被取消")
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} 异常: {e}")
                self.queue.task_done()

    async def _process_upload(self, task: UploadTask) -> None:
        if task.task_type == "fetch":
            await self._process_fetch_task(task)
            return
        max_retries = int(os.environ.get("UPLOAD_MAX_RETRIES", 4))
        for attempt in range(1, max_retries + 1):
            async with self._lock:
                if task.task_id not in self.active_tasks.get(task.user_id, {}):
                    return
            success = await self._try_upload_once(task, attempt, max_retries)
            if success:
                return
            if attempt < max_retries:
                # 网络错误（如上传超时）重试前退避，避免瞬时连发 4 次仍超时
                backoff = min(2 ** (attempt - 1), 30)
                logger.info(f"任务 {task.task_id[:8]} 第 {attempt}/{max_retries} 次失败，{backoff}s 后重试")
                await asyncio.sleep(backoff)
                if not await self._retry_parse_url(task):
                    break

    async def _retry_parse_url(self, task: UploadTask) -> bool:
        try:
            logger.info(f"任务 {task.task_id[:8]} 正在重新解析 URL: {task.parsed_content.url}")
            from ...provider.bilibili import BilibiliProvider

            provider = BilibiliProvider()
            results = await provider.parse([task.parsed_content.url], _get_constraints())
            if results and not isinstance(results[0], Exception):
                task.parsed_content = results[0]
                return True
            logger.error(f"任务 {task.task_id[:8]} 重新解析失败")
            return False
        except Exception as e:
            logger.exception(f"任务 {task.task_id[:8]} 重新解析时发生异常: {e}")
            return False

    async def _try_upload_once(self, task: UploadTask, attempt: int, max_retries: int) -> bool:
        f = task.parsed_content
        message = task.message
        medias = []
        try:
            async with RedisCache().lock(f.url, timeout=2 * CACHES_TIMER["LOCK"]):
                if not f.media or not f.media.urls:
                    from .bot import format_caption_for_telegram

                    await message.reply_text(format_caption_for_telegram(f, _get_constraints()))
                    return True

                media, mediathumb = await get_media_for_content(f)
                if not media:
                    if mediathumb and f.media.type not in ["video", "audio"]:
                        media = [mediathumb]
                    else:
                        from .bot import format_caption_for_telegram

                        await message.reply_text(format_caption_for_telegram(f, _get_constraints()))
                        return True

                if media:
                    medias.extend(media)
                if mediathumb:
                    medias.append(mediathumb)
                task.media = media
                task.mediathumb = mediathumb

                await self._upload_media(task)
                await self._try_delete_share_message(task)
                logger.info(f"任务 {task.task_id[:8]} 上传成功 (尝试 {attempt}/{max_retries})")
                return True

        except (BadRequest, RetryAfter, NetworkError, httpx.HTTPError) as err:
            should_retry = await self._handle_upload_error(err, task, attempt, max_retries, medias)
            return not should_retry
        except Exception as err:
            logger.exception(f"任务 {task.task_id[:8]} 第 {attempt}/{max_retries} 次未预期异常: {err}")
            cleanup_medias(medias)
            return False
        finally:
            cleanup_medias(medias)

    async def _handle_upload_error(self, err, task, attempt, max_retries, medias) -> bool:
        f = task.parsed_content
        message = task.message
        if isinstance(err, BadRequest):
            if (
                "Not enough rights to send" in err.message
                or "Need administrator rights in the channel chat" in err.message
            ):
                await message.chat.leave()
                cleanup_medias(medias)
                return False
            if any(x in err.message for x in ["Topic_deleted", "Topic_closed", "Message thread not found"]):
                cleanup_medias(medias)
                return False
            logger.error(f"任务 {task.task_id[:8]} 第 {attempt}/{max_retries} 次上传失败 (BadRequest): {err}")
            if f.media:
                f.media.need_download = True
            cleanup_medias(medias)
            return True
        if isinstance(err, RetryAfter):
            cleanup_medias(medias)
            await asyncio.sleep(err.retry_after)
            return True
        if isinstance(err, (NetworkError, httpx.HTTPError)):
            logger.error(f"任务 {task.task_id[:8]} 第 {attempt}/{max_retries} 次网络错误: {err}")
            cleanup_medias(medias)
            return True
        return False

    async def _upload_media(self, task: UploadTask) -> Any:
        f = task.parsed_content
        message = task.message
        media = task.media
        mediathumb = task.mediathumb

        from .bot import format_caption_for_telegram

        caption = format_caption_for_telegram(f, _get_constraints())

        if not media or not f.media:
            await message.reply_text(caption)
            return None

        if f.media.type == "video":
            result = await message.reply_video(
                media[0],
                caption=caption,
                supports_streaming=True,
                thumbnail=mediathumb,
                duration=f.media.duration,
                filename=f.media.filenames[0] if f.media.filenames else None,
                width=f.media.dimension.get("width", 0),
                height=f.media.dimension.get("height", 0),
            )
        elif f.media.type == "audio":
            result = await message.reply_audio(
                media[0],
                caption=caption,
                duration=f.media.duration,
                performer=f.author.name,
                thumbnail=mediathumb,
                title=f.media.title,
                filename=f.media.filenames[0] if f.media.filenames else None,
            )
        elif len(f.media.urls) == 1:
            if ".gif" in f.media.urls[0]:
                result = await message.reply_animation(
                    media[0],
                    caption=caption,
                    filename=f.media.filenames[0] if f.media.filenames else None,
                )
            else:
                result = await message.reply_photo(
                    media[0],
                    caption=caption,
                    filename=f.media.filenames[0] if f.media.filenames else None,
                )
        else:
            result = await self._upload_media_group(message, f, media, mediathumb, caption)

        await self._cache_upload_result(f, result)
        return result

    async def _upload_media_group(
        self, message: Message, f: ParsedContent, media: list, mediathumb: Any, caption: str
    ) -> tuple:
        # Telegram media group 单组上限 10 张。切割超长图后总数可能远超 10，
        # 只取前 10 张发送一个组（不再拆成多组连发），避免一条动态刷屏。
        limit = 10
        if len(media) > limit:
            logger.info(f"动态媒体 {len(media)} 项超过单组上限，仅发送前 {limit} 项")
        sub_media = media[:limit]
        sub_urls = f.media.urls[:limit]
        sub_fns = f.media.filenames[:limit]
        # caption 只挂在媒体组第一项，Telegram 会把它显示在相册下方作为整组说明；
        # 不再额外发独立文字消息，避免媒体与说明分成两条（与单图 reply_photo 行为一致）。
        return await message.reply_media_group(
            [
                (
                    InputMediaVideo(
                        img,
                        caption=caption if i == 0 else None,
                        filename=fn,
                        supports_streaming=True,
                    )
                    if ".gif" in mu
                    else InputMediaPhoto(img, caption=caption if i == 0 else None, filename=fn)
                )
                for i, (img, mu, fn) in enumerate(zip(sub_media, sub_urls, sub_fns, strict=False))
            ],
        )

    async def _cache_upload_result(self, f: ParsedContent, result: Any) -> None:
        if result is None or not f.media or not f.media.filenames:
            return
        urls = f.media.urls
        if isinstance(result, tuple):
            for idx, (filename, item) in enumerate(zip(f.media.filenames, result, strict=False)):
                url = urls[idx] if idx < len(urls) else None
                kind = _send_kind(f.media.type, url)
                attachment = item.effective_attachment
                if isinstance(attachment, tuple):
                    await cache_media(filename, attachment[0], kind)
                else:
                    await cache_media(filename, attachment, kind)
        else:
            kind = _send_kind(f.media.type, urls[0] if urls else None)
            attachment = result.effective_attachment
            if isinstance(attachment, tuple):
                await cache_media(f.media.filenames[0], attachment[0], kind)
            else:
                await cache_media(f.media.filenames[0], attachment, kind)

    async def _process_fetch_task(self, task: UploadTask) -> None:
        f = task.parsed_content
        message = task.message
        no_media = task.fetch_mode == "cover"

        from .bot import format_caption_for_telegram

        caption = format_caption_for_telegram(f, _get_constraints())

        async with RedisCache().lock(f.url, timeout=CACHES_TIMER["LOCK"]):
            if not f.media or not f.media.urls:
                return
            medias = []
            try:
                medias, mediathumb = await get_media_for_content(
                    f, compression=False, media_check_ignore=True, no_media=no_media
                )
                if mediathumb:
                    medias.insert(0, mediathumb)
                    mediafilenames = [f.media.thumbnail_filename, *f.media.filenames]
                else:
                    mediafilenames = f.media.filenames

                if len(medias) == 1:
                    result = await message.reply_document(
                        document=medias[0],
                        caption=caption,
                        filename=mediafilenames[0],
                    )
                    await cache_media(mediafilenames[0], result.effective_attachment, "document")
                else:
                    if len(medias) <= 10:
                        splits = [(medias, mediafilenames)]
                    else:
                        mid = len(medias) // 2
                        splits = [
                            (medias[:mid], mediafilenames[:mid]),
                            (medias[mid:], mediafilenames[mid:]),
                        ]
                    result = ()
                    for sub_m, sub_fn in splits:
                        sub_result = await message.reply_media_group(
                            [InputMediaDocument(m, filename=fn) for m, fn in zip(sub_m, sub_fn, strict=False)],
                        )
                        result += sub_result
                    await message.reply_text(caption)
                    for filename, item in zip(mediafilenames, result, strict=False):
                        attachment = item.effective_attachment
                        if isinstance(attachment, tuple):
                            await cache_media(filename, attachment[0], "document")
                        else:
                            await cache_media(filename, attachment, "document")
            except Exception as err:
                logger.exception(f"fetch 任务失败: {err} - {f.url}")
            finally:
                cleanup_medias(medias)

    async def _try_delete_share_message(self, task: UploadTask) -> None:
        message = task.message
        urls = task.urls
        try:
            if (
                len(urls) == 1
                and message.chat.type != ChatType.CHANNEL
                and not message.reply_to_message
                and message.text is not None
                and not message.is_automatic_forward
            ):
                match = re.match(BILIBILI_SHARE_URL_REGEX, message.text)
                if urls[0] == message.text or (match and match.group(0) == message.text):
                    await message.delete()
        except Exception as e:
            logger.debug(f"无法删除消息: {e}")

    async def start_workers(self) -> None:
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        logger.info(f"启动了 {self.max_workers} 个上传 Worker")

    async def stop_workers(self) -> None:
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("所有上传 Worker 已停止")
