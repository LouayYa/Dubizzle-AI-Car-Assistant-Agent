"""Photo-URL normalization + renderability.

The dubizzle CDN transcodes .heic uploads to WebP, so those URLs must be marked
renderable; a bare .heic on another host must not.
"""
from __future__ import annotations

from app.media import is_renderable_photo, normalize_photo_url

CDN = "https://dbz-images.dubizzle.com/images/2024/05/19/abc-"


def test_cdn_heic_is_renderable():
    assert is_renderable_photo(CDN + ".heic?impolicy=dpv") is True


def test_cdn_heic_without_param_gets_normalized_and_renderable():
    url = normalize_photo_url(CDN + ".heic")
    assert url.endswith("?impolicy=dpv")
    assert is_renderable_photo(url) is True


def test_cdn_jpeg_is_renderable():
    assert is_renderable_photo(CDN + ".jpeg?imwidth=800") is True


def test_non_cdn_heic_not_renderable():
    assert is_renderable_photo("https://example.com/photo.heic") is False


def test_non_cdn_jpg_renderable():
    assert is_renderable_photo("https://example.com/photo.jpg") is True


def test_empty_and_none():
    assert is_renderable_photo("") is False
    assert is_renderable_photo(None) is False
    assert normalize_photo_url(None) is None


def test_normalize_leaves_working_urls_unchanged():
    u = CDN + ".jpeg?imwidth=800"
    assert normalize_photo_url(u) == u
