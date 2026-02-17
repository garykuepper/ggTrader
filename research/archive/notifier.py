# discord_notifier.py
import io
import asyncio
import os
from dotenv import load_dotenv

import discord
import pandas as pd
from tabulate import tabulate


class Notifier:
    """
    Notifier that connects with discord.py and can send:
      - Plain text messages
      - Pandas DataFrames pretty-printed with tabulate
      - Falls back to CSV if too large
    """

    MAX_MESSAGE_LEN = 1900
    CSV_FALLBACK_CHAR_THRESHOLD = 8000

    def __init__(self, token: str, channel_id: int):
        self.token = token
        self.channel_id = int(channel_id)
        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)
        self._ready_evt = asyncio.Event()
        self._channel = None

        @self.client.event
        async def on_ready():
            ch = self.client.get_channel(self.channel_id)
            if ch is None:
                ch = await self.client.fetch_channel(self.channel_id)
            self._channel = ch
            self._ready_evt.set()
            print(f"✅ Logged in as {self.client.user} (channel: {ch.name})")

    async def start(self):
        """Login and wait until ready."""
        await self.client.login(self.token)
        asyncio.create_task(self.client.connect(reconnect=True))
        await self._ready_evt.wait()

    async def stop(self):
        """Close connection gracefully."""
        await self.client.close()

    async def send_text(self, content: str):
        """Send plain text message (splits if too long)."""
        for chunk in self._chunk_text(content, self.MAX_MESSAGE_LEN):
            await self._channel.send(chunk)

    async def send_dataframe(
        self,
        df: pd.DataFrame,
        title: str = "DataFrame",
        tablefmt: str = "github",
        index: bool = False,
    ):
        """Send DataFrame as tabulated text, or upload CSV if too large."""
        table_str = tabulate(df, headers="keys", tablefmt=tablefmt, showindex=index)
        message = f"**{title}**\n```\n{table_str}\n```"

        if len(message) <= self.MAX_MESSAGE_LEN:
            await self._channel.send(message)
        elif len(message) > self.CSV_FALLBACK_CHAR_THRESHOLD:
            buf = io.StringIO()
            df.to_csv(buf, index=index)
            buf.seek(0)
            await self._channel.send(
                content=f"{title} is too large to display, uploading as CSV.",
                file=discord.File(io.BytesIO(buf.getvalue().encode()), filename="table.csv"),
            )
        else:
            chunks = self._chunk_text(table_str, self.MAX_MESSAGE_LEN - 20)
            for i, chunk in enumerate(chunks, 1):
                await self._channel.send(f"{title} (part {i}/{len(chunks)})\n```\n{chunk}\n```")

    @staticmethod
    def _chunk_text(text: str, max_len: int):
        """Split text into chunks <= max_len, preferring line breaks."""
        lines, chunks, current = text.splitlines(True), [], ""
        for line in lines:
            if len(current) + len(line) <= max_len:
                current += line
            else:
                chunks.append(current)
                current = line
        if current:
            chunks.append(current)
        return chunks


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    load_dotenv()  # load .env file

    TOKEN = os.getenv("DISCORD_TOKEN")
    CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

    async def main():
        df = pd.DataFrame(
            [
                {"pair": "BTC", "signal": "BUY",  "entry": 27850, "sl": 27300, "tp": 29000},
                {"pair": "ETH", "signal": "SELL", "entry": 1750,  "sl": 1790,  "tp": 1680},
                {"pair": "XRP", "signal": "SELL", "entry": 0.3,   "sl": 0.25,  "tp": 0.35},
            ]
        )
        timestamp = pd.Timestamp.now().tz_localize('America/Los_Angeles')
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S %p %Z%z")
        notifier = Notifier(TOKEN, CHANNEL_ID)
        await notifier.start()
        # await notifier.send_text(f"ggTrader is live ✅")
        await notifier.send_dataframe(df, title=f"Current Signals [{ts_str}]", tablefmt="github", index=False)
        await notifier.stop()

    asyncio.run(main())
