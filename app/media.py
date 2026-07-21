"""Photo-URL helpers.

The dubizzle CDN transcodes `.heic` uploads to WebP on the fly if a policy
param is present — renaming the extension instead just 404s, since the CDN
only has the exact stored filename. Non-CDN `.heic`/`.heif` links can't be
fixed, so they're flagged non-renderable and the UI shows a "View photo"
link instead of a broken image.
"""
from __future__ import annotations

from urllib.parse import urlsplit

# Hosts that transcode via the CDN.
_CDN_HOSTS = ("dbz-images.dubizzle.com", "dubizzle.com")
# Extensions that need CDN transcoding to render.
_NON_WEB_EXT = (".heic", ".heif", ".tif", ".tiff")


def _host(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower()
    except ValueError:
        return ""


def _path_ext(url: str) -> str:
    path = urlsplit(url).path.lower()
    dot = path.rfind(".")
    return path[dot:] if dot != -1 else ""


def _is_cdn(url: str) -> bool:
    host = _host(url)
    return any(host == h or host.endswith("." + h) for h in _CDN_HOSTS)


def normalize_photo_url(url: str | None) -> str | None:
    """Append the CDN transcoding param to bare `.heic`/`.heif` CDN links."""
    if not url or not isinstance(url, str):
        return url
    url = url.strip()
    if _is_cdn(url) and _path_ext(url) in _NON_WEB_EXT and "?" not in url:
        return url + "?impolicy=dpv"
    return url


def is_renderable_photo(url: str | None) -> bool:
    """True if the URL can be shown inline (browser <img> / st.image)."""
    if not url or not isinstance(url, str):
        return False
    if not url.lower().startswith(("http://", "https://")):
        return False
    if _is_cdn(url):
        return True  # CDN transcodes .heic/.heif -> WebP
    return _path_ext(url) not in _NON_WEB_EXT
