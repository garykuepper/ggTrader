"""Notification abstractions for live trading alerts and daily reports.

Currently supports Telegram (via Bot API) and Discord (via webhooks). Both
backends use only ``requests`` (already a project dependency) — no extra
libraries like python-telegram-bot or discord.py needed.

Tokens are loaded from environment variables. If the relevant env vars are
missing, the notifier silently no-ops so reports still generate locally.

Usage:
    from ggTrader.utils.notifier import build_notifiers_from_env

    notifiers = build_notifiers_from_env()
    for n in notifiers:
        n.send_text("Daily PnL: +$12.34")
        n.send_photo(Path("results/reports/daily.png"), caption="Equity curve")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("ggTraderLive")


class BaseNotifier:
    """Notifier interface. Subclasses implement send_text and send_photo."""

    name: str = "base"

    def send_text(self, message: str) -> bool:
        raise NotImplementedError

    def send_photo(self, image_path: Path, caption: str = "") -> bool:
        raise NotImplementedError


class TelegramNotifier(BaseNotifier):
    """Sends messages via the Telegram Bot API.

    Setup:
      1. Talk to @BotFather on Telegram to create a bot — get the bot token.
      2. Add the bot to your chat (or message it directly).
      3. Find your chat_id by visiting:
         https://api.telegram.org/bot<TOKEN>/getUpdates
         after sending a message to the bot. Look for "chat":{"id":...}.
      4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
    """

    name = "telegram"
    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, bot_token: str, chat_id: str, parse_mode: str = "Markdown") -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self._base = self.BASE_URL.format(token=bot_token)

    def send_text(self, message: str) -> bool:
        """Send a text message. Telegram limit is 4096 chars per message."""
        if len(message) > 4000:
            # Split into chunks at line boundaries
            chunks = self._split_message(message, 4000)
            return all(self._post_text(c) for c in chunks)
        return self._post_text(message)

    def _split_message(self, message: str, max_len: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in message.split("\n"):
            line_len = len(line) + 1
            if current_len + line_len > max_len and current:
                chunks.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _post_text(self, message: str) -> bool:
        try:
            payload: dict = {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": True,
            }
            # Only include parse_mode if explicitly set — empty string causes
            # Telegram to fall back to its default parser, which ignores our intent.
            if self.parse_mode:
                payload["parse_mode"] = self.parse_mode
            r = requests.post(
                f"{self._base}/sendMessage",
                json=payload,
                timeout=15,
            )
            if not r.ok:
                logger.warning(
                    f"  [Telegram] sendMessage failed ({r.status_code}): {r.text[:200]}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"  [Telegram] sendMessage exception: {e!r}")
            return False

    def send_photo(self, image_path: Path, caption: str = "") -> bool:
        if not image_path.exists():
            logger.warning(f"  [Telegram] photo not found: {image_path}")
            return False
        try:
            data: dict = {
                "chat_id": self.chat_id,
                "caption": caption[:1024],
            }
            if self.parse_mode:
                data["parse_mode"] = self.parse_mode
            with open(image_path, "rb") as f:
                r = requests.post(
                    f"{self._base}/sendPhoto",
                    data=data,
                    files={"photo": f},
                    timeout=30,
                )
            if not r.ok:
                logger.warning(
                    f"  [Telegram] sendPhoto failed ({r.status_code}): {r.text[:200]}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"  [Telegram] sendPhoto exception: {e!r}")
            return False


class DiscordNotifier(BaseNotifier):
    """Sends messages via a Discord webhook.

    Setup:
      1. In your Discord server, edit a channel -> Integrations -> Webhooks
         -> New Webhook -> Copy URL.
      2. Set DISCORD_WEBHOOK_URL in .env.
    """

    name = "discord"

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send_text(self, message: str) -> bool:
        if len(message) > 1900:
            chunks = self._split_message(message, 1900)
            return all(self._post(content=c) for c in chunks)
        return self._post(content=message)

    def _split_message(self, message: str, max_len: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in message.split("\n"):
            line_len = len(line) + 1
            if current_len + line_len > max_len and current:
                chunks.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _post(self, content: str = "", file_path: Optional[Path] = None) -> bool:
        try:
            if file_path and file_path.exists():
                with open(file_path, "rb") as f:
                    r = requests.post(
                        self.webhook_url,
                        data={"content": content},
                        files={"file": (file_path.name, f)},
                        timeout=30,
                    )
            else:
                r = requests.post(
                    self.webhook_url,
                    json={"content": content},
                    timeout=15,
                )
            if not r.ok:
                logger.warning(
                    f"  [Discord] webhook failed ({r.status_code}): {r.text[:200]}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"  [Discord] webhook exception: {e!r}")
            return False

    def send_photo(self, image_path: Path, caption: str = "") -> bool:
        return self._post(content=caption, file_path=image_path)


def build_notifiers_from_env() -> list[BaseNotifier]:
    """Construct all notifier backends with credentials available in env vars.

    Silently skips any backend missing required credentials. Returns an empty
    list if no notifiers are configured (caller should treat this as "local
    output only" — not an error).
    """
    notifiers: list[BaseNotifier] = []

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if tg_token and tg_chat:
        notifiers.append(TelegramNotifier(tg_token, tg_chat))
    elif tg_token or tg_chat:
        logger.warning(
            "  [Notifier] Telegram partially configured: need both "
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
        )

    discord_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if discord_url:
        notifiers.append(DiscordNotifier(discord_url))

    if not notifiers:
        logger.info(
            "  [Notifier] No notification channels configured "
            "(set TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID or DISCORD_WEBHOOK_URL)"
        )
    else:
        logger.info(
            f"  [Notifier] Configured channels: {', '.join(n.name for n in notifiers)}"
        )
    return notifiers
