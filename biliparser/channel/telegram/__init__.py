import os

from ...model import MediaConstraints, ParsedContent
from ...model import PreparedMedia as PreparedMedia
from ...provider import ProviderRegistry
from ...utils import logger
from .. import Channel

TELEGRAM_UPLOAD_SIZE = 50 * 1024 * 1024
TELEGRAM_UPLOAD_SIZE_LOCAL = 2 * 1024 * 1024 * 1024
TELEGRAM_CAPTION_LENGTH = 1024


class TelegramChannel(Channel):
    def __init__(self):
        self._local_mode = bool(os.environ.get("LOCAL_MODE", False))
        self._registry: ProviderRegistry | None = None

    @property
    def media_constraints(self) -> MediaConstraints:
        return MediaConstraints(
            max_upload_size=(TELEGRAM_UPLOAD_SIZE_LOCAL if self._local_mode else TELEGRAM_UPLOAD_SIZE),
            max_download_size=TELEGRAM_UPLOAD_SIZE_LOCAL,
            caption_max_length=TELEGRAM_CAPTION_LENGTH,
            local_mode=self._local_mode,
        )

    def format_caption(self, content: ParsedContent) -> str:
        from .bot import format_caption_for_telegram

        return format_caption_for_telegram(content, self.media_constraints)

    async def send_content(self, content, media, context):
        pass

    async def send_text(self, text, context):
        pass

    async def cache_sent_media(self, content: ParsedContent, result) -> None:
        pass

    async def get_cached_media(self, filename: str) -> str | None:
        # 通用接口默认按 photo 命名空间读取（file_id 与媒体类型强绑定，见 uploader._cache_key）
        from .uploader import get_cached_media_file_id

        return await get_cached_media_file_id(filename, "photo")

    async def start(self, provider_registry: ProviderRegistry) -> None:
        self._registry = provider_registry
        logger.info("TelegramChannel starting...")

    async def stop(self) -> None:
        logger.info("TelegramChannel stopped")
