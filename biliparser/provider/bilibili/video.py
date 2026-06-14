import datetime
import math
import os
import re
from difflib import SequenceMatcher
from functools import cached_property
from urllib.parse import parse_qs, urlparse

import orjson
from bilibili_api import video

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

_DEFAULT_MAX_SIZE = 50 * 1024 * 1024  # 50MB


class Video(Feed):
    cidcontent: dict = {}
    epcontent: dict = {}
    infocontent: dict = {}
    page = 1
    quality = video.VideoQuality._8K
    quality_explicit: bool = False  # 用户是否显式指定了画质（/video <画质>），指定时不自动降档
    _dash_size_exhausted: bool = False  # DASH 所有画质均超过降档阈值时置位，供 handle() 回退封面
    reply_type: int = 1

    def extract_episode_info(self, target: str):
        if not self.epid or not self.epcontent or not self.epcontent.get("result"):
            return None
        for episode in self.epcontent["result"].get("episodes"):
            if str(episode.get("id")) == str(self.epid):
                return episode.get(target)
        for subsection in self.epcontent["result"].get("section"):
            for episode in subsection.get("episodes"):
                if str(episode.get("id")) == str(self.epid):
                    return episode.get(target)

    @cached_property
    def cid(self):
        if self.infocontent and self.infocontent.get("data"):
            if self.page != 1 and self.infocontent["data"].get("pages"):
                for item in self.infocontent["data"]["pages"]:
                    if item.get("page") == self.page:
                        return item.get("cid")
            self.page = 1
            return self.infocontent["data"].get("cid")
        return self.extract_episode_info("cid")

    @cached_property
    def bvid(self):
        if self.infocontent and self.infocontent.get("data"):
            return self.infocontent["data"].get("bvid")
        return self.extract_episode_info("bvid")

    @cached_property
    def aid(self):
        if self.infocontent and self.infocontent.get("data"):
            return self.infocontent["data"].get("aid")
        return self.extract_episode_info("aid")

    @cached_property
    def epid(self):
        if self.epcontent and self.epcontent.get("result") and self.epcontent["result"].get("episodes"):
            return self.epcontent["result"]["episodes"][-1].get("id")

    @cached_property
    def ssid(self):
        if self.epcontent and self.epcontent.get("result"):
            return self.epcontent["result"].get("season_id")

    @cached_property
    def url(self):
        return f"https://www.bilibili.com/video/av{self.aid}?p={self.page}"

    @property
    def cache_key(self):
        return {
            "bangumi:ep": f"bangumi:ep:{self.epid}",
            "bangumi:ss": f"bangumi:ss:{self.ssid}",
            "video:aid": f"video:aid:{self.aid}",
            "video:bvid": f"video:bvid:{self.bvid}",
        }

    def set_quality(self, in_str: str | None):
        if in_str is None:
            return
        in_str = in_str.strip().upper().replace("+", "PLUS")
        similarities = [
            (opt, SequenceMatcher(lambda x: x == "_", in_str, opt).ratio())
            for opt in [q.name for q in video.VideoQuality]
        ]
        best_match = max(similarities, key=lambda x: x[1])
        self.quality = video.VideoQuality[best_match[0]]
        self.quality_explicit = True

    def clear_cached_properties(self):
        for key in ["epid", "ssid", "aid", "bvid", "cid"]:
            if hasattr(self, key) and getattr(self, key) is None:
                delattr(self, key)

    async def __get_video_result(self, qn: video.VideoQuality, max_size: int):
        params = {"avid": self.aid, "cid": self.cid}
        if qn:
            params["qn"] = qn.value
        r = await bili_api_request(
            self.client,
            "/x/player/playurl",
            params=params,
            cookies=(await credentialFactory.get()).get_cookies(),
        )
        video_result = r.json()
        size_limit = int(os.environ.get("VIDEO_SIZE_LIMIT", max_size))
        if (
            video_result.get("code") == 0
            and video_result.get("data")
            and video_result.get("data").get("durl")
            and video_result.get("data").get("durl")[0].get("size") < size_limit
        ):
            url = video_result["data"]["durl"][0]["url"]
            result, url = await self.test_url_status_code(url, self.url)
            if not result and video_result["data"]["durl"][0].get("backup_url", None):
                backup_urls = video_result["data"]["durl"][0]["backup_url"]
                for item in backup_urls:
                    url = item
                    result, item = await self.test_url_status_code(item, self.url)
                    if result:
                        break
            if result:
                self.mediacontent = video_result
                self.mediaduration = round(video_result["data"]["durl"][0]["length"] / 1000)
                self.mediaurls = url
                self.mediafallbackurl = url
                self.mediatype = "video"
                self.mediaraws = False
                self.mediafilesize = video_result.get("data").get("durl")[0].get("size")
                return True

    async def __get_dash_video(self, max_size: int):
        params = {
            "avid": self.aid,
            "cid": self.cid,
            "qn": 125,
            "fnver": 0,
            "fnval": 4048,
            "fourk": 1,
            "voice_balance": 1,
        }
        r = await bili_api_request(
            self.client,
            "/x/player/playurl",
            params=params,
            cookies=(await credentialFactory.get()).get_cookies(),
        )
        video_result = r.json()
        if video_result.get("code") != 0 or not video_result.get("data"):
            logger.error(f"获取Dash视频流错误: {video_result}")
            return False
        dash_data = video_result["data"]
        ## TODO: rewrite self VideoDownloadURLDataDetecter with built-in test_url_status_code
        detecter = video.VideoDownloadURLDataDetecter(data=dash_data)
        streams = detecter.detect(
            video_min_quality=video.VideoQuality._360P,
            video_max_quality=self.quality,
            codecs=[video.VideoCodecs(os.environ.get("VIDEO_CODEC", "avc"))],
        )  # 可以设置成hev/av01减少文件体积，但是tg不二压会造成部分老设备直接解码指定codec时不展示，需要指定成avc
        video_streams = [video_stream for video_stream in streams if type(video_stream) is video.VideoStreamDownloadURL]
        audio_streams = [audio_stream for audio_stream in streams if type(audio_stream) is video.AudioStreamDownloadURL]
        if not video_streams or not audio_streams:
            logger.error(f"获取Dash视频流错误: {streams}")
            return False
        video_streams.sort(key=lambda x: x.video_quality.value, reverse=True)
        audio_streams.sort(key=lambda x: x.audio_quality.value, reverse=True)
        dash_info = dash_data.get("dash", {})
        # 使用实际流分辨率而非原始上传分辨率，避免 Telegram 尺寸不匹配
        stream_dimensions = {}
        for rv in dash_info.get("video", []):
            stream_dimensions[(rv.get("id"), rv.get("codecs", ""))] = (rv.get("width", 0), rv.get("height", 0))
        audio_url = None
        audio_size = 0
        for audio_stream in audio_streams:
            audio_size, audio_stream.url = await self.test_url_status_code(audio_stream.url, self.url)
            if audio_size:
                audio_url = audio_stream.url
                break
        if not audio_url:
            logger.error(f"无可用Dash视频音频流清晰度: {streams}")
            return False
        dash_duration = dash_info.get("duration", 0)
        hard_limit = int(os.environ.get("VIDEO_SIZE_LIMIT", max_size))
        if self.quality_explicit:
            # 显式指定画质：不自动降档，只取不超过请求档的最高画质
            # detect() 可能返回高于 video_max_quality 的流（如 DOLBY=126），需按 value 上限过滤
            capped = [s for s in video_streams if s.video_quality.value <= self.quality.value]
            if not capped:
                # 没有任何流满足上限（极少见），退回到 detect 的最低档
                capped = sorted(video_streams, key=lambda x: x.video_quality.value)[:1]
            video_stream = capped[0]
            video_size, video_stream.url = await self.test_url_status_code(video_stream.url, self.url)
            if not video_size:
                logger.error(f"无可用Dash视频流清晰度: {streams}")
                return False
            total_size = audio_size + video_size
            if total_size >= hard_limit:
                raise ParserException(
                    f"视频文件过大：{video_stream.video_quality.name} 约 {self.human_size(total_size)}，"
                    f"超过大小限制 {self.human_size(hard_limit)}，请指定更低画质后重试",
                    self.url,
                )
            logger.info(f"选择Dash视频清晰度(指定):{video_stream.video_quality.name} 大小:{video_size}")
            self.__commit_dash_media(video_stream, audio_url, total_size, dash_duration, stream_dimensions)
            return True
        # 自动模式：从高到低逐档检查体积，选取首个不超过降档阈值的画质
        # VIDEO_DOWNGRADE_SIZE 未配置时回退到 VIDEO_SIZE_LIMIT，保持旧行为
        downgrade_env = os.environ.get("VIDEO_DOWNGRADE_SIZE")
        downgrade_limit = int(downgrade_env) if downgrade_env else hard_limit
        for video_stream in video_streams:
            video_size, video_stream.url = await self.test_url_status_code(video_stream.url, self.url)
            if audio_size and video_size and (audio_size + video_size < downgrade_limit):
                logger.info(f"选择Dash视频清晰度:{video_stream.video_quality.name} 大小:{video_size}")
                self.__commit_dash_media(
                    video_stream, audio_url, audio_size + video_size, dash_duration, stream_dimensions
                )
                return True
        # 所有画质（含最低档）均超过降档阈值。仅当用户显式配置了 VIDEO_DOWNGRADE_SIZE
        # 才回退封面（新行为）；否则保持旧行为（保留已有的 MP4 直链或封面，不强制覆盖）。
        if downgrade_env:
            logger.warning(f"所有Dash画质均超过降档阈值 {self.human_size(downgrade_limit)}，回退封面: {self.url}")
            self._dash_size_exhausted = True
        else:
            logger.error(f"无可用Dash视频流清晰度: {streams}")
        return False

    def __commit_dash_media(self, video_stream, audio_url, total_size, dash_duration, stream_dimensions):
        """将选定的 DASH 视频/音频轨写入媒体字段，并使用实际流分辨率设置尺寸。"""
        self.mediaurls = [video_stream.url, audio_url]
        self.mediatype = "video"
        self.mediaraws = True
        self.mediamerge = True
        self.mediafilesize = total_size
        if dash_duration:
            self.mediaduration = dash_duration
        # bilibili-api 17.4.1 部分流 video_codecs 为 None，需空值保护，否则崩溃导致尺寸残留为原始上传分辨率
        codec_val = video_stream.video_codecs.value if video_stream.video_codecs else ""
        w, h = next(
            (
                (w, h)
                for (qid, codecs), (w, h) in stream_dimensions.items()
                if qid == video_stream.video_quality.value and (not codec_val or codecs.startswith(codec_val))
            ),
            (0, 0),
        )
        if w and h:
            self.mediadimention = {"width": w, "height": h, "rotate": self.mediadimention.get("rotate", 0)}

    @staticmethod
    def human_size(num: int) -> str:
        size = float(num)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024 or unit == "GB":
                return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}{unit}"
            size /= 1024
        return f"{num}B"

    async def handle(self, constraints=None, extra: dict | None = None) -> "Video":
        max_size = constraints.max_upload_size if constraints else _DEFAULT_MAX_SIZE
        caption_length = constraints.caption_max_length if constraints else 1024
        if extra:
            self.set_quality(extra.get("quality"))
        logger.info(f"处理视频信息: 链接: {self.rawurl}")
        match = re.search(
            r"(?:bilibili\.com(?:/video|/bangumi/play)?|b23\.tv|acg\.tv)/(?:(?P<bvid>BV\w{10})|av(?P<aid>\d+)|ep(?P<epid>\d+)|ss(?P<ssid>\d+)|)",
            self.rawurl,
        )
        match_fes = re.search(
            r"bilibili\.com/festival/(?P<festivalid>\w+)",
            self.rawurl,
        )
        pr = urlparse(self.rawurl)
        qs = parse_qs(pr.query)
        seek_id = None
        if "comment_secondary_id" in qs:
            seek_id = qs["comment_secondary_id"][0]
        elif "comment_root_id" in qs:
            seek_id = qs["comment_root_id"][0]
        elif pr.fragment.startswith("reply"):
            seek_id = pr.fragment.removeprefix("reply")
        if match_fes:
            if "bvid" in qs:
                __bvid = qs["bvid"][0]
                params = {"bvid": __bvid}
                self.bvid = __bvid
            else:
                raise ParserException("视频链接解析错误", self.rawurl)
        elif match:
            __bvid = match.group("bvid")
            __epid = match.group("epid")
            __aid = match.group("aid")
            __ssid = match.group("ssid")
            if "p" in qs:
                __page = qs["p"][0]
                if __page.isdigit():
                    self.page = max(1, int(__page))
            if __epid:
                params = {"ep_id": __epid}
                self.epid = __epid
            elif __bvid:
                params = {"bvid": __bvid}
                self.bvid = __bvid
            elif __aid:
                params = {"aid": __aid}
                self.aid = __aid
            elif __ssid:
                params = {"season_id": __ssid}
                self.ssid = __ssid
            else:
                raise ParserException("视频链接解析错误", self.rawurl)
            if self.epid is not None or self.ssid is not None:
                # 1.获取缓存
                try:
                    cache = (
                        await RedisCache().get(self.cache_key["bangumi:ep"])
                        if self.epid
                        else await RedisCache().get(self.cache_key["bangumi:ss"])
                    )
                except Exception as e:
                    logger.exception(f"拉取番剧缓存错误: {e}")
                    cache = None
                # 2.拉取番剧
                self.clear_cached_properties()
                if cache:
                    self.epcontent = orjson.loads(cache)  # type: ignore
                    logger.info(f"拉取番剧缓存:epid {self.epid}" if self.epid else f"拉取番剧缓存:ssid {self.ssid}")
                else:
                    try:
                        r = await bili_api_request(
                            self.client,
                            "/pgc/view/web/season",
                            params=params,
                        )
                        self.epcontent = r.json()
                    except Exception as e:
                        raise ParserException(
                            f"番剧获取错误:{self.epid or self.ssid}",
                            self.rawurl,
                            e,
                        )
                    # 3.番剧解析
                    if not self.epcontent or not self.epcontent.get("result"):
                        # Anime detects non-China IP
                        raise ParserException(
                            f"番剧解析错误:{self.epid or self.ssid} {self.epcontent}",
                            self.rawurl,
                            self.epcontent,
                        )
                    self.clear_cached_properties()
                    if not self.epid or not self.ssid or not self.aid:
                        raise ParserException(
                            f"番剧解析错误:{self.epid} {self.ssid} {self.aid}",
                            self.rawurl,
                            self.epcontent,
                        )
                    # 4.缓存评论
                    try:
                        for key in [
                            self.cache_key["bangumi:ep"],
                            self.cache_key["bangumi:ss"],
                        ]:
                            await RedisCache().set(
                                key,
                                orjson.dumps(self.epcontent),
                                ex=CACHES_TIMER["BANGUMI"],
                                nx=True,
                            )
                    except Exception as e:
                        logger.exception(f"缓存番剧错误: {e}")
                params = {"aid": self.aid}
        else:
            raise ParserException("视频链接解析错误", self.rawurl)
        # 1.获取缓存
        try:
            cache = (
                await RedisCache().get(self.cache_key["video:aid"])
                if self.aid
                else await RedisCache().get(self.cache_key["video:bvid"])
            )
        except Exception as e:
            logger.exception(f"拉取视频缓存错误: {e}")
            cache = None
        # 2.拉取视频
        self.clear_cached_properties()
        if cache:
            self.infocontent = orjson.loads(cache)  # type: ignore
            logger.info(f"拉取视频缓存:{self.aid or self.bvid}")
        else:
            try:
                r = await bili_api_request(
                    self.client,
                    "/x/web-interface/view",
                    params=params,
                )
                self.infocontent = r.json()
            except Exception as e:
                raise ParserException(
                    f"视频获取错误:{self.aid or self.bvid}",
                    self.rawurl,
                    e,
                )
            # 3.视频解析
            if not self.infocontent or not self.infocontent.get("data"):
                # Video detects non-China IP
                raise ParserException(
                    f"视频解析错误{self.aid or self.bvid}",
                    r.url,
                    self.infocontent,
                )
            if not self.aid or not self.bvid or not self.cid:
                raise ParserException(
                    f"视频解析错误:{self.aid} {self.bvid} {self.cid}",
                    self.rawurl,
                    self.epcontent,
                )
            # 4.缓存视频
            try:
                for key in [self.cache_key["video:aid"], self.cache_key["video:bvid"]]:
                    await RedisCache().set(
                        key,
                        orjson.dumps(self.infocontent),
                        ex=CACHES_TIMER["VIDEO"],
                        nx=True,
                    )
            except Exception as e:
                logger.exception(f"缓存视频错误: {e}")
        detail = self.infocontent["data"]
        self.user = detail.get("owner").get("name")
        self.uid = detail.get("owner").get("mid")
        content = "发布视频"
        if detail.get("tname"):
            content += f"-{detail.get('tname')}"
        if detail.get("tname_v2"):
            content += f"-{detail.get('tname_v2')}"
        content += "\n"
        if detail.get("pages") and len(detail["pages"]) > 1:
            content += f"第{self.page}P/共{len(detail['pages'])}P\n"
        if detail.get("stat"):
            if detail.get("stat").get("now_rank"):
                content += f"当前排行榜第{detail.get('stat').get('now_rank')}位\n"
            elif detail.get("stat").get("his_rank"):
                content += f"历史排行榜第{detail.get('stat').get('his_rank')}位\n"
            content += f"播放量:{self.wan(detail.get('stat').get('view', 0))} 弹幕:{self.wan(detail.get('stat').get('danmaku', 0))} 评论:{self.wan(detail.get('stat').get('reply', 0))}\n"
            content += f"点赞:{self.wan(detail.get('stat').get('like', 0))} 投币:{self.wan(detail.get('stat').get('coin', 0))} 收藏:{self.wan(detail.get('stat').get('favorite', 0))} 转发:{self.wan(detail.get('stat').get('share', 0))}\n"
        if detail.get("pubdate"):
            content += (
                f"发布日期:{datetime.datetime.fromtimestamp(detail.get('pubdate')).strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        if detail.get("ctime") and detail.get("ctime") != detail.get("pubdate"):
            content += (
                f"上传日期:{datetime.datetime.fromtimestamp(detail.get('ctime')).strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        self.content = content
        self.extra_markdown = f"[{escape_markdown(detail.get('title'))}]({self.url})"
        extra_desc = detail.get("desc") or detail.get("dynamic")
        if extra_desc and extra_desc != "-":
            extra_desc = (
                f"\n**>{escape_markdown(detail.get('desc') or detail.get('dynamic')).replace(chr(10), chr(10) + '>')}||"
            )
        if extra_desc and extra_desc != "-" and len(self.extra_markdown + extra_desc) < caption_length:
            self.extra_markdown += extra_desc
        self.mediatitle = detail.get("title")
        self.mediaurls = detail.get("pic")
        self.mediathumb = detail.get("pic")
        self.mediadimention = detail.get("pages")[self.page - 1].get("dimension")
        ## 标准化 width+height不超过10k https://github.com/tdlib/td/blob/5c77c4692c28eb48a68ef1c1eeb1b1d732d507d3/td/telegram/PhotoSize.cpp#L422
        if self.mediadimention is not None and (
            self.mediadimention.get("width", 0) + self.mediadimention.get("height", 0) > 10000
        ):
            scale = 10000 / (self.mediadimention.get("width", 0) + self.mediadimention.get("height", 0))
            self.mediadimention = {
                "width": math.floor(scale * self.mediadimention.get("width", 0)),
                "height": math.floor(scale * self.mediadimention.get("height", 0)),
                "rotate": self.mediadimention.get("rotate", 0),
            }
        self.mediatype = "image"
        cover_url = detail.get("pic")
        self.replycontent = await self.parse_reply(self.aid, self.reply_type, seek_id)
        try:
            for mp4_qn in [
                video.VideoQuality._720P,
                video.VideoQuality._480P,
                video.VideoQuality._360P,
            ]:
                if await self.__get_video_result(mp4_qn, max_size):
                    break
            await self.__get_dash_video(max_size)
        except ParserException:
            # 显式指定画质且超过大小限制：直接向上抛出，提示用户文件过大
            raise
        except Exception as e:
            logger.exception(f"视频下载解析错误: {e}")
        # 自动降档模式下所有画质均超阈值：强制回退封面，清空视频相关字段
        if self._dash_size_exhausted:
            logger.info(f"回退发送封面: {self.url}")
            self.mediaurls = cover_url
            self.mediathumb = cover_url
            self.mediatype = "image"
            self.mediaraws = False
            self.mediamerge = False
            self.mediafallbackurl = ""
            self.mediafilesize = 0
        return self
