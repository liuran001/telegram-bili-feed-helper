"""
biliparser.uploader — 平台无关的上传共享层

公共 API：
  download: get_media, handle_dash_media, get_media_for_content, cleanup_medias
  queue:    UploadTask, UploadQueueManager
"""

from .download import (
    CacheKeyBuilder,
    CacheLookup,
    cleanup_medias,
    expand_long_images,
    get_media,
    get_media_for_content,
    handle_dash_media,
    split_long_image_bytes,
)
from .queue import UploadQueueManager, UploadTask

__all__ = [
    "CacheKeyBuilder",
    "CacheLookup",
    "cleanup_medias",
    "expand_long_images",
    "get_media",
    "get_media_for_content",
    "handle_dash_media",
    "split_long_image_bytes",
    "UploadQueueManager",
    "UploadTask",
]
