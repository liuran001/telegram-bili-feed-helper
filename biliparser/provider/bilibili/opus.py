import re
from functools import cached_property, lru_cache

import orjson

from ...storage.cache import RedisCache
from ...utils import (
    escape_markdown,
    logger,
)
from .api import (
    CACHES_TIMER,
    ParserException,
    bili_api_request,
)
from .credential import credentialFactory
from .feed import Feed

# web-dynamic v1 detail 需要的 features 参数，缺失会导致部分字段（如 opus 大图）拿不到
DYNAMIC_FEATURES = (
    "itemOpusStyle,opusBigCover,onlyfansVote,decorationCard,onlyfansAssetsV2,ugcDelete,editable,opusPrivateVisible"
)

# major.type -> major 子字段名
MAJOR_TYPE_KEY = {
    "MAJOR_TYPE_ARCHIVE": "archive",
    "MAJOR_TYPE_DRAW": "draw",
    "MAJOR_TYPE_OPUS": "opus",
    "MAJOR_TYPE_COURSES": "courses",
    "MAJOR_TYPE_PGC": "pgc",
    "MAJOR_TYPE_LIVE": "live",
    "MAJOR_TYPE_LIVE_RCMD": "live_rcmd",
    "MAJOR_TYPE_ARTICLE": "article",
    "MAJOR_TYPE_MUSIC": "music",
    "MAJOR_TYPE_COMMON": "common",
    "MAJOR_TYPE_UGC_SEASON": "ugc_season",
    "MAJOR_TYPE_MEDIALIST": "medialist",
}


class Opus(Feed):
    detailcontent: dict = {}
    dynamic_id: int = 0
    user: str = ""
    __content: str = ""
    forward_user: str = ""
    forward_uid: int = 0
    forward_content: str = ""
    stat_content: str = ""
    has_forward: bool = False

    @cached_property
    def item(self) -> dict:
        return self.detailcontent.get("item", {}) if self.detailcontent else {}

    @cached_property
    def reply_type(self):
        # web-v1 basic.comment_type 直接就是评论接口所需的 type（动态=17，视频=1 等）
        return int(self.item.get("basic", {}).get("comment_type") or 17)

    @cached_property
    def rid(self):
        basic = self.item.get("basic", {})
        return int(basic.get("comment_id_str") or basic.get("rid_str") or self.dynamic_id)

    @property
    @lru_cache(maxsize=1)
    def content(self):
        content = self.__content
        if self.has_forward:
            if self.forward_user:
                content += f"//@{self.forward_user}:\n"
            content += self.forward_content
        if self.stat_content:
            content += "\n" + self.stat_content
        return self.shrink_line(content)

    @content.setter
    def content(self, content):
        self.__content = content

    @cached_property
    def content_markdown(self):
        content_markdown = escape_markdown(self.__content)
        if self.has_forward:
            if self.forward_uid:
                content_markdown += f"//{self.make_user_markdown(self.forward_user, self.forward_uid)}:\n"
            elif self.forward_user:
                content_markdown += f"//@{escape_markdown(self.forward_user)}:\n"
            content_markdown += escape_markdown(self.forward_content)
        if not content_markdown.endswith("\n"):
            content_markdown += "\n"
        content_markdown += self.stat_content
        return self.shrink_line(content_markdown)

    @cached_property
    def url(self):
        return f"https://t.bilibili.com/{self.dynamic_id}"

    @property
    def cache_key(self):
        return {"opus:dynamic_id": f"opus:dynamic_id:{self.dynamic_id}"}

    @staticmethod
    def __desc_text(module_dynamic: dict | None) -> str:
        """从 module_dynamic.desc 取已展平的文本（emoji 已是 [xxx] 文本形式）。"""
        if not module_dynamic:
            return ""
        desc = module_dynamic.get("desc")
        if not desc:
            return ""
        return desc.get("text") or ""

    @staticmethod
    def __normalize_url(url: str) -> str:
        if url.startswith("//"):
            return "https:" + url
        return url

    def __handle_major(self, major: dict | None) -> str:
        """处理 module_dynamic.major：设置媒体与外链，返回附加正文文本（如 opus 标题/正文）。"""
        if not major or not isinstance(major, dict):
            return ""
        mtype = major.get("type")
        key = MAJOR_TYPE_KEY.get(mtype)
        if not key:
            return ""
        sub = major.get(key)
        if not isinstance(sub, dict):
            return ""
        if mtype == "MAJOR_TYPE_DRAW":
            self.mediaurls = [item["src"] for item in sub.get("items", []) if item.get("src")]
            self.mediatype = "image"
            return ""
        if mtype == "MAJOR_TYPE_OPUS":
            pics = [p["url"] for p in sub.get("pics", []) if p.get("url")]
            if pics:
                self.mediaurls = pics
                self.mediatype = "image"
            text = ""
            if sub.get("title"):
                text += sub["title"] + "\n"
            summary = sub.get("summary") or {}
            if summary.get("text"):
                text += summary["text"]
            return text
        if mtype == "MAJOR_TYPE_ARCHIVE":
            if sub.get("cover"):
                self.mediaurls = sub["cover"]
                self.mediathumb = sub["cover"]
                self.mediatype = "image"
            bvid, aid, title = sub.get("bvid"), sub.get("aid"), sub.get("title")
            if title and (bvid or aid):
                link = f"https://www.bilibili.com/video/{bvid}" if bvid else f"https://www.bilibili.com/video/av{aid}"
                self.extra_markdown = f"[{escape_markdown(title)}]({link})"
            return ""
        # courses / pgc / live / article / music / common 等：取封面 + 标题外链
        if sub.get("cover"):
            self.mediaurls = sub["cover"]
            self.mediathumb = sub["cover"]
            self.mediatype = "image"
        if sub.get("title") and sub.get("jump_url"):
            self.extra_markdown = f"[{escape_markdown(sub['title'])}]({self.__normalize_url(sub['jump_url'])})"
        return ""

    def __handle_modules(self, modules: dict, *, is_forward_origin: bool) -> str:
        """解析一个 modules（dict），返回正文文本；设置作者/媒体。is_forward_origin 表示这是被转发的原动态。"""
        author = modules.get("module_author") or {}
        name = author.get("name", "")
        mid = author.get("mid", 0)
        module_dynamic = modules.get("module_dynamic") or {}
        text = self.__desc_text(module_dynamic)
        major_text = self.__handle_major(module_dynamic.get("major"))
        if major_text:
            text = f"{text}\n{major_text}" if text else major_text
        if is_forward_origin:
            self.forward_user = name
            self.forward_uid = mid
        else:
            self.user = name
            self.uid = mid
        return text

    async def handle(self, constraints=None):
        logger.info(f"处理动态信息: 链接: {self.rawurl}")
        match = re.search(
            r"(?:www|t|h|m)\.bilibili\.com\/(?:[^\/?]+\/)*?(\d+)(?:[\/?].*)?",
            self.rawurl,
        )
        if not match:
            raise ParserException("动态链接错误", self.rawurl)
        self.dynamic_id = int(match.group(1))
        # 1.获取缓存
        try:
            cache = await RedisCache().get(self.cache_key["opus:dynamic_id"])
        except Exception as e:
            logger.exception(f"拉取动态缓存错误: {e}")
            cache = None
        # 2.拉取动态
        if cache:
            self.detailcontent = orjson.loads(cache)  # type: ignore
            logger.info(f"拉取动态缓存: {self.dynamic_id}")
        else:
            try:
                # desktop/v1/detail 对 FORWARD 类型返回 -1024，改用通用 v1/detail（schema 不同）
                # 该接口未登录会返回 -352 风控错误，必须带上登录 Cookie
                r = await bili_api_request(
                    self.client,
                    "/x/polymer/web-dynamic/v1/detail",
                    params={"id": self.dynamic_id, "features": DYNAMIC_FEATURES},
                    cookies=(await credentialFactory.get()).get_cookies(),
                )
                response = r.json()
            except Exception as e:
                raise ParserException(f"动态获取错误:{self.dynamic_id}", self.rawurl, e)
            # 3.动态解析
            if not response or not response.get("data") or not response["data"].get("item"):
                raise ParserException("动态解析错误", self.rawurl, response)
            self.detailcontent = response["data"]
            # 4.缓存动态
            try:
                await RedisCache().set(
                    self.cache_key["opus:dynamic_id"],
                    orjson.dumps(self.detailcontent),
                    ex=CACHES_TIMER["OPUS"],
                    nx=True,
                )
            except Exception as e:
                logger.exception(f"缓存动态错误: {e}")
        item = self.item
        modules = item.get("modules") or {}
        # 主动态（转发时为转发者及其附言）
        self.content = self.__handle_modules(modules, is_forward_origin=False)
        # 转发：原动态在 item.orig
        orig = item.get("orig")
        if item.get("type") == "DYNAMIC_TYPE_FORWARD" and isinstance(orig, dict):
            self.has_forward = True
            orig_modules = orig.get("modules") or {}
            self.forward_content = self.__handle_modules(orig_modules, is_forward_origin=True)
        # 统计信息取主动态的 module_stat
        module_stat = modules.get("module_stat") or {}
        if module_stat:
            self.stat_content = (
                f"转发:{self.wan(module_stat.get('forward', {}).get('count', 0))} "
                f"评论:{self.wan(module_stat.get('comment', {}).get('count', 0))} "
                f"点赞:{self.wan(module_stat.get('like', {}).get('count', 0))}"
            )
        self.extra_markdown = f"[{escape_markdown(self.user)}的动态]({self.url})"
        self.replycontent = await self.parse_reply(self.rid, self.reply_type)
        return self
