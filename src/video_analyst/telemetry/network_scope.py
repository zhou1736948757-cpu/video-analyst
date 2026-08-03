"""Network scope classification for telemetry.

将一次 HTTP 调用分类为：

* ``none``：未发送 HTTP（Mock provider / cache hit / Dry Run）
* ``local_mock``：访问了 localhost / 127.0.0.1 / ::1 / MockTransport / 被
  monkeypatched 的 transport；不视为真实外部 API
* ``external_provider``：访问了非本地 hostname；视为真实外部 API

``classify_network_scope`` 是判定 ``real_vision_api_used`` 的关键：

* 只看 URL hostname，不看 provider 名称（避免被 provider 名 "anthropic" 误导）
* transport_metadata（``MockTransport`` / monkeypatched transport）会强制
  归类为 ``local_mock``
* cache hit / dry run 调用方不需要 HTTP，由调用方在 record 里直接写
  ``network_scope="none"``，不进本函数
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from video_analyst.telemetry.models import NetworkScope


_LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
    }
)


def _is_local_host(host: str) -> bool:
    """判断 hostname 是否指向本地回环。"""
    if not host:
        return False
    lowered = host.lower().strip("[]")
    if lowered in _LOCAL_HOSTNAMES:
        return True
    # 127.0.0.0/8, ::1, 0.0.0.0
    try:
        ip = ipaddress.ip_address(lowered)
        return bool(
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass
    # host:port -> host
    if ":" in lowered and not lowered.startswith("["):
        lowered = lowered.split(":", 1)[0]
    try:
        ip = ipaddress.ip_address(lowered)
        return bool(
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def _resolve_host_if_needed(host: str) -> bool:
    """对形如真实外部域名的字符串，做一次正向解析；解析到非本地 IP 视为外部。"""
    if not host or _is_local_host(host):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return True  # 解析失败时保守地视为外部（不是 mock）
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return True
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
        return True
    return True


def classify_network_scope(
    url: str | None,
    transport_metadata: dict[str, Any] | None = None,
) -> tuple[NetworkScope, str | None]:
    """返回 ``(network_scope, endpoint_host)``。

    * ``url``：完整 HTTP URL；只取 hostname，不保留 path / query / fragment
    * ``transport_metadata``：可选，传入 ``{"transport": "mock"}`` 等
      强制归类为 ``local_mock``

    优先级：

    1. transport_metadata 标记 mock → ``local_mock``
    2. url 是本地回环 → ``local_mock``
    3. url 是外部域名 → ``external_provider``
    4. 没有 url（无 HTTP） → ``none``
    """
    if transport_metadata:
        transport = str(transport_metadata.get("transport", "")).lower()
        if transport in {"mock", "monkeypatch", "monkeypatched", "mocktransport"}:
            return ("local_mock", None)
    if not url:
        return ("none", None)
    try:
        parsed = urlparse(url)
    except Exception:
        return ("external_provider", None)
    host = parsed.hostname or ""
    if not host:
        return ("external_provider", None)
    if _is_local_host(host):
        return ("local_mock", host)
    if _resolve_host_if_needed(host):
        return ("external_provider", host)
    return ("external_provider", host)


__all__ = ["classify_network_scope"]