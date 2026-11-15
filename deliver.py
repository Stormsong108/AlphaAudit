"""飞书投递：用**本 profile（小金）自己的 app 凭据**直连飞书 Open API。

为什么不用 lark-cli：
lark-cli 的 bot 身份是另一个 app（cli_aa8d5dc8…），
而 config 里的 open_id 属于小金 app（cli_a959e68d…），
open_id 是 app 维度的 —— 跨 app 使用会报 `open_id cross app`。
所以这里直接用 ~/.hermes/profiles/xiaojin/.env 里的凭据取 tenant_access_token。

投递失败会抛异常，调用方负责降级，绝不静默吞掉。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = "https://open.feishu.cn/open-apis"
ENV_PATH = Path.home() / ".hermes" / "profiles" / "xiaojin" / ".env"

_token_cache: dict[str, float] = {}


def _read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_PATH.exists():
        return env
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def get_token() -> str:
    """取 tenant_access_token，缓存 25 分钟（官方有效期 2 小时）。"""
    now = time.time()
    cached = _token_cache.get("token")
    if cached and now - _token_cache.get("ts", 0) < 1500:
        return cached
    env = _read_env()
    app_id = env.get("FEISHU_APP_ID")
    app_secret = env.get("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        raise RuntimeError(f"缺少飞书凭据，检查 {ENV_PATH}")
    req = urllib.request.Request(
        f"{API}/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    token = body.get("tenant_access_token")
    if not token:
        raise RuntimeError(f"取 token 失败: {body}")
    _token_cache["token"] = token
    _token_cache["ts"] = now
    return token


def send_text(text: str, *, receive_id: str, id_type: str = "chat_id") -> dict:
    """发纯文本消息。返回带 message_id 的凭据。"""
    token = get_token()
    payload = {
        "receive_id": receive_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False),
    }
    req = urllib.request.Request(
        f"{API}/im/v1/messages?receive_id_type={id_type}",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"飞书 HTTP {exc.code}: {detail}") from exc
    if body.get("code") != 0:
        raise RuntimeError(f"飞书返回错误: {body}")
    data = body.get("data") or {}
    return {
        "ok": True,
        "message_id": data.get("message_id"),
        "chat_id": data.get("chat_id"),
    }


def upload_file(path: str | Path, *, file_name: str | None = None, file_type: str = "stream") -> str:
    """上传文件到飞书云空间，返回 file_key。

    注意：文件消息（msg_type=file）必须先上传拿 file_key，再发消息；
    不能把本地路径直接塞进消息体。
    """
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"待上传文件不存在: {p}")
    payload_bytes = p.read_bytes()
    name = file_name or p.name
    boundary = "----HermesBoundary" + uuid.uuid4().hex
    parts: list[bytes] = []

    def _field(key: str, value: str) -> None:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )

    _field("file_type", file_type)
    _field("file_name", name)
    parts.append(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
    )
    parts.append(payload_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    token = get_token()
    req = urllib.request.Request(
        f"{API}/im/v1/files",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"飞书上传文件失败 HTTP {exc.code}: {detail}") from exc
    if payload.get("code") != 0:
        raise RuntimeError(f"飞书上传文件返回错误: {payload}")
    file_key = (payload.get("data") or {}).get("file_key")
    if not file_key:
        raise RuntimeError(f"飞书未返回 file_key: {payload}")
    return file_key


def send_file(file_key: str, *, receive_id: str, id_type: str = "chat_id") -> dict:
    """发送文件消息（msg_type=file）。"""
    token = get_token()
    payload = {
        "receive_id": receive_id,
        "msg_type": "file",
        "content": json.dumps({"file_key": file_key}),
    }
    req = urllib.request.Request(
        f"{API}/im/v1/messages?receive_id_type={id_type}",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"飞书 HTTP {exc.code}: {detail}") from exc
    if body.get("code") != 0:
        raise RuntimeError(f"飞书返回错误: {body}")
    data = body.get("data") or {}
    return {
        "ok": True,
        "message_id": data.get("message_id"),
        "chat_id": data.get("chat_id"),
        "file_key": file_key,
    }
