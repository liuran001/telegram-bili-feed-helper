from types import SimpleNamespace

import pytest
from bilibili_api import video

from biliparser.provider.bilibili.video import _dash_stream_candidates, _resolve_video_codec


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
