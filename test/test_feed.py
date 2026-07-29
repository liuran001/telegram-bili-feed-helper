"""测试 provider/bilibili/feed.py — Feed 基类的纯数据属性和工具方法"""

import httpx
import pytest

from biliparser.provider.bilibili.feed import Feed, expand_upos_urls


class ConcreteFeed(Feed):
    """Feed 是 ABC，需要一个具体子类来测试"""

    async def handle(self):
        return self


class TestFeedStaticMethods:
    def test_shrink_line_normal(self):
        assert Feed.shrink_line("  hello  ") == "hello"

    def test_shrink_line_empty(self):
        assert Feed.shrink_line("") == ""
        assert Feed.shrink_line(None) == ""

    def test_clean_cn_tag_style(self):
        result = Feed.clean_cn_tag_style("\\#标签\\#")
        assert "\\#标签 " in result

    def test_clean_cn_tag_style_empty(self):
        assert Feed.clean_cn_tag_style("") == ""
        assert Feed.clean_cn_tag_style(None) == ""

    def test_wan_below_threshold(self):
        assert Feed.wan(9999) == 9999

    def test_wan_above_threshold(self):
        result = Feed.wan(10000)
        assert "万" in str(result)
        assert "1.00" in str(result)

    def test_wan_large_number(self):
        result = Feed.wan(1234567)
        assert "万" in str(result)

    def test_make_user_markdown(self):
        result = Feed.make_user_markdown("用户名", "12345")
        assert "用户名" in result
        assert "12345" in result
        assert "space.bilibili.com" in result

    def test_make_user_markdown_empty(self):
        assert Feed.make_user_markdown("", "") == ""
        assert Feed.make_user_markdown("user", "") == ""
        assert Feed.make_user_markdown("", "123") == ""


class TestFeedProperties:
    @pytest.fixture
    def feed(self):
        client = httpx.AsyncClient()
        return ConcreteFeed("https://test.bilibili.com/123", client)

    def test_default_values(self, feed):
        assert feed.user == ""
        assert feed.uid == ""
        assert feed.mediatype == ""
        assert feed.mediaduration == 0
        assert feed.mediatitle == ""
        assert feed.extra_markdown == ""

    def test_content_property(self, feed):
        feed.content = "  test content  "
        assert feed.content == "test content"

    def test_mediaurls_single(self, feed):
        feed.mediaurls = "https://example.com/image.jpg"
        assert feed.mediaurls == ["https://example.com/image.jpg"]

    def test_mediaurls_list(self, feed):
        feed.mediaurls = ["https://a.jpg", "https://b.jpg"]
        assert len(feed.mediaurls) == 2

    def test_mediafilename(self, feed):
        feed.mediaurls = ["https://example.com/path/image.jpg"]
        assert feed.mediafilename == ["image.jpg"]

    def test_mediathumb(self, feed):
        feed.mediathumb = "https://example.com/thumb.jpg"
        assert feed.mediathumb == "https://example.com/thumb.jpg"
        assert feed.mediathumbfilename == "thumb.jpg"

    def test_mediathumb_empty(self, feed):
        assert feed.mediathumb == ""
        assert feed.mediathumbfilename == ""

    def test_url_default(self, feed):
        assert feed.url == "https://test.bilibili.com/123"

    def test_cache_key_default(self, feed):
        assert feed.cache_key == {}

    def test_rawurl(self, feed):
        assert feed.rawurl == "https://test.bilibili.com/123"


class TestUposMirrorExpansion:
    """UPOS_DOMAIN 镜像展开 — B站对境外 IP 只返回 ov/akam 节点，需把境内镜像补进候选池"""

    def test_no_upos_domain_keeps_source_urls(self, monkeypatch):
        monkeypatch.delenv("UPOS_DOMAIN", raising=False)
        urls = ["https://upos-sz-mirrorcosov.bilivideo.com/upgcxcode/a.m4s?sig=1"]
        assert expand_upos_urls(urls) == urls

    def test_every_configured_mirror_enters_candidates(self, monkeypatch):
        """所有配置的镜像都要进候选池，而不是只进随机一个"""
        monkeypatch.setenv("UPOS_DOMAIN", "m1.bilivideo.com, m2.bilivideo.com ,m3.bilivideo.com")
        source = "https://upos-sz-mirrorcosov.bilivideo.com/upgcxcode/a.m4s?sig=1"
        result = expand_upos_urls([source])
        hosts = [httpx.URL(url).host for url in result]
        assert hosts == [
            "m1.bilivideo.com",
            "m2.bilivideo.com",
            "m3.bilivideo.com",
            "upos-sz-mirrorcosov.bilivideo.com",
        ]

    def test_path_and_query_are_preserved(self, monkeypatch):
        """UPOS 签名在 query 里，替换域名时必须原样保留 path+query"""
        monkeypatch.setenv("UPOS_DOMAIN", "m1.bilivideo.com")
        source = "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/66/18/x-1-30064.m4s?e=ig8&upsig=abc&oi=123"
        assert expand_upos_urls([source])[0] == (
            "https://m1.bilivideo.com/upgcxcode/66/18/x-1-30064.m4s?e=ig8&upsig=abc&oi=123"
        )

    def test_native_urls_are_kept_as_fallback(self, monkeypatch):
        """境内镜像可能对某些资源 404，原生 URL 必须保留兜底"""
        monkeypatch.setenv("UPOS_DOMAIN", "m1.bilivideo.com")
        native = [
            "https://upos-sz-mirrorcosov.bilivideo.com/upgcxcode/a.m4s?sig=1",
            "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/a.m4s?sig=1",
        ]
        result = expand_upos_urls(native)
        assert native[0] in result and native[1] in result

    def test_duplicates_are_collapsed(self, monkeypatch):
        """镜像域名与原生域名相同时不产生重复候选"""
        monkeypatch.setenv("UPOS_DOMAIN", "upos-sz-mirrorcosov.bilivideo.com")
        source = "https://upos-sz-mirrorcosov.bilivideo.com/upgcxcode/a.m4s?sig=1"
        assert expand_upos_urls([source]) == [source]

    def test_empty_and_blank_domains_are_ignored(self, monkeypatch):
        monkeypatch.setenv("UPOS_DOMAIN", " , ,")
        urls = ["https://upos-sz-mirrorcosov.bilivideo.com/a.m4s"]
        assert expand_upos_urls(urls) == urls
