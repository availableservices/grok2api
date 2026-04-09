"""
Reverse interface: Imagine WebSocket image stream.
"""

import asyncio
import base64
import orjson
import re
import time
import uuid
from typing import AsyncGenerator, Dict, Optional

import aiohttp

from app.core.config import get_config
from app.core.logger import logger
from app.services.reverse.utils.headers import build_ws_headers, build_sso_cookie
from app.services.reverse.utils.websocket import WebSocketClient

WS_IMAGINE_URL = "wss://grok.com/ws/imagine/listen"


class _BlockedError(Exception):
    pass


class ImagineWebSocketReverse:
    """Imagine WebSocket reverse interface."""

    def __init__(self) -> None:
        self._url_pattern = re.compile(r"/images/([a-f0-9-]+)\.(png|jpg|jpeg)")
        self._client = WebSocketClient()

    def _parse_image_url(self, url: str) -> tuple[Optional[str], Optional[str]]:
        match = self._url_pattern.search(url or "")
        if not match:
            return None, None
        return match.group(1), match.group(2).lower()

    def _is_final_image(self, url: str, blob_size: int, final_min_bytes: int) -> bool:
        # Final image must satisfy byte-size threshold to avoid tiny preview
        # images being treated as final outputs.
        return blob_size >= final_min_bytes

    def _classify_image(self, url: str, blob: str, final_min_bytes: int, medium_min_bytes: int) -> Optional[Dict[str, object]]:
        if not url or not blob:
            return None

        image_id, ext = self._parse_image_url(url)
        image_id = image_id or uuid.uuid4().hex
        blob_size = len(blob)
        is_final = self._is_final_image(url, blob_size, final_min_bytes)

        stage = (
            "final"
            if is_final
            else ("medium" if blob_size > medium_min_bytes else "preview")
        )

        return {
            "type": "image",
            "image_id": image_id,
            "ext": ext,
            "stage": stage,
            "blob": blob,
            "blob_size": blob_size,
            "url": url,
            "is_final": is_final,
        }

    def _build_request_message(self, request_id: str, prompt: str, aspect_ratio: str, enable_nsfw: bool, model_name: Optional[str] = None) -> Dict[str, object]:
        properties: Dict[str, object] = {
            "section_count": 0,
            "is_kids_mode": False,
            "enable_nsfw": enable_nsfw,
            "skip_upsampler": False,
            "enable_side_by_side": True,
            "is_initial": False,
            "aspect_ratio": aspect_ratio,
        }
        if model_name == "imagine-x-1":
            properties["enable_pro"] = True
        if model_name:
            properties["model_name"] = model_name
        return {
            "type": "conversation.item.create",
            "timestamp": int(time.time() * 1000),
            "item": {
                "type": "message",
                "content": [
                    {
                        "requestId": request_id,
                        "text": prompt,
                        "type": "input_text",
                        "properties": properties,
                    }
                ],
            },
        }

    async def _fetch_blob_from_cdn(self, url: str, token: str) -> str:
        """Fetch image from CDN URL and return as base64 data URI."""
        if url.startswith("/"):
            url = f"https://assets.grok.com{url}"
        try:
            user_agent = get_config("proxy.user_agent") or ""
            headers = {
                "Cookie": build_sso_cookie(token),
                "User-Agent": user_agent,
                "Accept": "image/*,*/*",
                "Referer": "https://grok.com/",
                "Cache-Control": "no-cache",
            }
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning(f"CDN fetch failed: {resp.status} for {url}")
                        return ""
                    content = await resp.read()
                    ct = resp.content_type or "image/jpeg"
                    b64 = base64.b64encode(content).decode()
                    return f"data:{ct};base64,{b64}"
        except Exception as e:
            logger.warning(f"CDN fetch error for {url}: {e}")
            return ""

    async def stream(
        self,
        token: str,
        prompt: str,
        aspect_ratio: str = "2:3",
        n: int = 1,
        enable_nsfw: bool = True,
        max_retries: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, object], None]:
        retries = max(1, max_retries if max_retries is not None else 1)
        parallel_enabled = bool(get_config("image.blocked_parallel_enabled", True))
        logger.info(
            f"Image generation: prompt='{prompt[:50]}...', n={n}, ratio={aspect_ratio}, nsfw={enable_nsfw}"
        )

        async def _collect_once() -> list[Dict[str, object]]:
            items: list[Dict[str, object]] = []
            async for item in self._stream_once(
                token, prompt, aspect_ratio, n, enable_nsfw, model_name
            ):
                items.append(item)
            return items

        for attempt in range(retries):
            try:
                items = await _collect_once()
                for item in items:
                    yield item
                return
            except _BlockedError:
                retries_left = retries - (attempt + 1)
                if retries_left > 0 and parallel_enabled:
                    logger.warning(
                        f"WebSocket blocked/reviewed, launching {retries_left} parallel retries"
                    )
                    tasks = [asyncio.create_task(_collect_once()) for _ in range(retries_left)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            continue
                        has_final = any(
                            isinstance(item, dict)
                            and item.get("type") == "image"
                            and item.get("is_final")
                            for item in result
                        )
                        if has_final:
                            for item in result:
                                yield item
                            return
                    yield {
                        "type": "error",
                        "error_code": "blocked",
                        "error": "blocked_no_final_image",
                        "parallel_attempts": retries_left,
                    }
                    return
                if attempt + 1 < retries:
                    logger.warning(
                        f"WebSocket blocked/reviewed, retry {attempt + 1}/{retries}"
                    )
                    continue
                yield {
                    "type": "error",
                    "error_code": "blocked",
                    "error": "blocked_no_final_image",
                }
                return
            except Exception as e:
                logger.error(f"WebSocket stream failed: {e}")
                yield {
                    "type": "error",
                    "error_code": "ws_stream_failed",
                    "error": str(e),
                }
                return

    async def _stream_once(
        self,
        token: str,
        prompt: str,
        aspect_ratio: str,
        n: int,
        enable_nsfw: bool,
        model_name: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, object], None]:
        request_id = str(uuid.uuid4())
        headers = build_ws_headers(token=token)
        timeout = float(get_config("image.timeout"))
        stream_timeout = float(get_config("image.stream_timeout"))
        final_timeout = float(get_config("image.final_timeout"))
        blocked_grace_cfg = get_config("image.blocked_grace_seconds")
        blocked_grace = float(blocked_grace_cfg) if blocked_grace_cfg is not None else 10.0
        blocked_grace = max(1.0, min(blocked_grace, final_timeout))
        final_min_bytes = int(get_config("image.final_min_bytes"))
        medium_min_bytes = int(get_config("image.medium_min_bytes"))

        try:
            conn = await self._client.connect(
                WS_IMAGINE_URL,
                headers=headers,
                timeout=timeout,
                ws_kwargs={
                    "heartbeat": 20,
                    "receive_timeout": stream_timeout,
                },
            )
        except Exception as e:
            status = getattr(e, "status", None)
            error_code = (
                "rate_limit_exceeded" if status == 429 else "connection_failed"
            )
            logger.error(f"WebSocket connect failed: {e}")
            yield {
                "type": "error",
                "error_code": error_code,
                "status": status,
                "error": str(e),
            }
            return

        try:
            async with conn as ws:
                message = self._build_request_message(
                    request_id, prompt, aspect_ratio, enable_nsfw, model_name
                )
                await ws.send_json(message)
                logger.info(f"WebSocket request sent: {prompt[:80]}...")

                final_ids: set[str] = set()
                completed = 0
                start_time = last_activity = time.monotonic()
                medium_received_time: Optional[float] = None
                # For ws_only models (e.g. imagine-x-1): buffer latest blob per image_id,
                # only finalize when type:"json" completion signal arrives.
                pending_blobs: Dict[str, Dict[str, object]] = {}

                while time.monotonic() - start_time < timeout:
                    try:
                        ws_msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    except asyncio.TimeoutError:
                        now = time.monotonic()
                        if (
                            medium_received_time
                            and completed == 0
                            and now - medium_received_time > blocked_grace
                        ):
                            logger.warning(
                                "Imagine stream blocked suspected: received medium preview but no valid final image "
                                f"within {blocked_grace:.1f}s (request_id={request_id})"
                            )
                            raise _BlockedError()
                        if completed > 0 and now - last_activity > 10:
                            logger.info(
                                f"WebSocket idle timeout, collected {completed} images"
                            )
                            break
                        continue

                    if ws_msg.type == aiohttp.WSMsgType.TEXT:
                        last_activity = time.monotonic()
                        try:
                            msg = orjson.loads(ws_msg.data)
                        except orjson.JSONDecodeError as e:
                            logger.warning(f"WebSocket message decode failed: {e}")
                            continue

                        msg_type = msg.get("type")

                        if msg_type == "image":
                            info = self._classify_image(
                                msg.get("url", ""),
                                msg.get("blob", ""),
                                final_min_bytes,
                                medium_min_bytes,
                            )
                            if not info:
                                continue

                            image_id = info["image_id"]
                            if info["stage"] == "medium" and medium_received_time is None:
                                medium_received_time = time.monotonic()

                            if model_name:
                                # For ws_only models: never finalize by blob size alone.
                                # Buffer the latest blob; finalize only on type:"json" completion.
                                info["is_final"] = False
                                if info["stage"] == "final":
                                    info["stage"] = "medium"
                                pending_blobs[image_id] = info
                                yield info
                            else:
                                if info["is_final"] and image_id not in final_ids:
                                    final_ids.add(image_id)
                                    completed += 1
                                    logger.debug(
                                        f"Final image received: id={image_id}, size={info['blob_size']}"
                                    )
                                yield info

                        elif msg_type == "error":
                            logger.warning(
                                f"WebSocket error: {msg.get('err_code', '')} - {msg.get('err_msg', '')}"
                            )
                            yield {
                                "type": "error",
                                "error_code": msg.get("err_code", ""),
                                "error": msg.get("err_msg", ""),
                            }
                            return

                        elif msg_type == "json":
                            # imagine-x-1 sends progress/completion as type:"json"
                            status = str(msg.get("current_status") or "")
                            pct = float(msg.get("percentage_complete") or 0.0)
                            image_id = str(msg.get("image_id") or msg.get("job_id") or "")
                            blob = msg.get("blob", "")
                            url = msg.get("url", "")

                            logger.debug(
                                f"imagine-x-1 progress: status={status}, "
                                f"pct={pct:.1f}%, image_id={image_id}"
                            )

                            # Mark that generation has started (for blocked detection)
                            if pct > 0 and medium_received_time is None:
                                medium_received_time = time.monotonic()

                            is_complete = (
                                status in ("completed", "done", "finished") or pct >= 100.0
                            )

                            if blob or url:
                                # Image data directly in the JSON message
                                info = self._classify_image(url, blob, final_min_bytes, medium_min_bytes)
                                if info:
                                    if info["stage"] == "medium" and medium_received_time is None:
                                        medium_received_time = time.monotonic()
                                    if info["is_final"] and info["image_id"] not in final_ids:
                                        final_ids.add(info["image_id"])
                                        completed += 1
                                        logger.debug(
                                            f"Final image from JSON: id={info['image_id']}, "
                                            f"size={info['blob_size']}"
                                        )
                                    yield info
                            elif is_complete and image_id and image_id not in final_ids:
                                # Job complete — finalize from buffered blob if available
                                if image_id in pending_blobs:
                                    final_info = dict(pending_blobs.pop(image_id))
                                    final_info["is_final"] = True
                                    final_info["stage"] = "final"
                                    final_ids.add(image_id)
                                    completed += 1
                                    logger.debug(
                                        f"Final image from buffer: id={image_id}, "
                                        f"size={final_info['blob_size']}"
                                    )
                                    yield final_info
                                else:
                                    # No buffered blob — fallback to CDN
                                    asset_url = f"https://assets.grok.com/images/{image_id}.jpg"
                                    logger.info(
                                        f"imagine-x-1 job completed, fetching from CDN: {asset_url}"
                                    )
                                    cdn_blob = await self._fetch_blob_from_cdn(asset_url, token)
                                    if cdn_blob:
                                        info = self._classify_image(asset_url, cdn_blob, 0, 0)
                                        if info:
                                            info["is_final"] = True
                                            info["stage"] = "final"
                                            final_ids.add(image_id)
                                            completed += 1
                                            logger.debug(
                                                f"Final image from CDN: id={image_id}, "
                                                f"size={info['blob_size']}"
                                            )
                                            yield info
                                    else:
                                        logger.warning(
                                            f"CDN fetch returned empty for image_id={image_id}"
                                        )

                        if completed >= n:
                            logger.info(f"WebSocket collected {completed} final images")
                            break

                        if (
                            medium_received_time
                            and completed == 0
                            and time.monotonic() - medium_received_time > final_timeout
                        ):
                            logger.warning(
                                "Imagine stream final-timeout suspected review/block: "
                                f"no final image reached threshold in {final_timeout:.1f}s "
                                f"(request_id={request_id})"
                            )
                            raise _BlockedError()

                    elif ws_msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        logger.warning(f"WebSocket closed/error: {ws_msg.type}")
                        yield {
                            "type": "error",
                            "error_code": "ws_closed",
                            "error": f"websocket closed: {ws_msg.type}",
                        }
                        break

        except aiohttp.ClientError as e:
            logger.error(f"WebSocket connection error: {e}")
            yield {"type": "error", "error_code": "connection_failed", "error": str(e)}


__all__ = ["ImagineWebSocketReverse", "WS_IMAGINE_URL"]
