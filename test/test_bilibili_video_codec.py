from types import SimpleNamespace

import httpx
import pytest
from bilibili_api import video

from biliparser.provider.bilibili.video import (
    _dash_stream_candidates,
    _prioritize_url,
    _resolve_video_codec,
)


@pytest.mark.parametrize(
    ("codec_name", "expected"),
    [
        ("avc", video.VideoCodecs.AVC),
        ("AVC", video.VideoCodecs.AVC),
        ("hev", video.VideoCodecs.HEV),
        ("hvc", video.VideoCodecs.HEV),
        ("av1", video.VideoCodecs.AV1),
        ("av01", video.VideoCodecs.AV1),
        ("", video.VideoCodecs.AVC),
    ],
)
def test_resolve_video_codec_supports_names_and_aliases(codec_name, expected):
    assert _resolve_video_codec(codec_name) is expected


def test_dash_stream_candidates_preserve_primary_and_backup_upos_urls():
    primary = "https://upos-primary.example/video.m4s?token=1"
    backup_camel = "https://upos-backup-a.example/video.m4s?token=2"
    backup_snake = "https://upos-backup-b.example/video.m4s?token=3"
    selected = SimpleNamespace(
        url=primary,
        video_quality=video.VideoQuality._4K,
        video_codecs=video.VideoCodecs.AVC,
    )
    dash_data = {
        "dash": {
            "video": [
                {
                    "id": video.VideoQuality._4K.value,
                    "codecs": "avc1.640034",
                    "baseUrl": primary,
                    "backupUrl": [backup_camel],
                    "backup_url": [backup_snake, backup_camel],
                }
            ]
        }
    }

    assert _dash_stream_candidates(dash_data, selected, "video") == [primary, backup_camel, backup_snake]


def test_prioritize_url_puts_selected_first_and_keeps_native_fallback(monkeypatch):
    """未配置 UPOS_DOMAIN 时保持原行为：选中的 URL 置顶，原生候选跟随其后"""
    monkeypatch.delenv("UPOS_DOMAIN", raising=False)
    native = [
        "https://upos-sz-mirrorcosov.bilivideo.com/upgcxcode/a.m4s?sig=1",
        "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/a.m4s?sig=1",
    ]
    assert _prioritize_url(native[1], native) == [native[1], native[0]]


def test_prioritize_url_expands_all_upos_mirrors_into_candidates(monkeypatch):
    """配置 UPOS_DOMAIN 后，所有镜像都要进候选池供下载层测速择优。

    修复前 test_url_status_code 只把随机命中的那 1 个镜像放进池子，
    下载层只能在「1 个境内镜像 + 2 个境外节点」里选，无法在多个境内镜像间比较。
    """
    monkeypatch.setenv("UPOS_DOMAIN", "m1.bilivideo.com,m2.bilivideo.com,m3.bilivideo.com")
    native = ["https://upos-sz-mirrorcosov.bilivideo.com/upgcxcode/a.m4s?sig=1"]
    selected = "https://m2.bilivideo.com/upgcxcode/a.m4s?sig=1"

    result = _prioritize_url(selected, native)

    assert result[0] == selected, "test_url_status_code 验证过的 URL 仍应置顶"
    hosts = {httpx.URL(url).host for url in result}
    assert {"m1.bilivideo.com", "m2.bilivideo.com", "m3.bilivideo.com"} <= hosts
    assert "upos-sz-mirrorcosov.bilivideo.com" in hosts, "原生节点保留兜底"
    assert len(result) == len(set(result)), "候选不得重复"
