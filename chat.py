#!/usr/bin/env python3
"""Two OpenRouter models talk in a console window. You can cut in at any time."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import RenderableType
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.visual import visualize
from textual.widget import Widget
from textual.widgets import Input, Static

# Used when the matching command-line flag is left off. Flags still override these.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LEFT_MODEL = "openai/gpt-4o-mini"
DEFAULT_RIGHT_MODEL = "openai/gpt-4o-mini"
DEFAULT_LEFT_NAME = "Ada"
DEFAULT_RIGHT_NAME = "Lin"
DEFAULT_OBSERVER_NAME = "Observer"
DEFAULT_REPLIES = -1  # per bot; -1 is unlimited
DEFAULT_STARTER = "left"  # "left" or "right"
DEFAULT_TOPIC = "Introduce yourselves, and get to know each other."
LEFT_PERSONA = "You are curious and warm, and you ask specific questions."
RIGHT_PERSONA = "You are concise and a little skeptical, and you want a concrete example."
DEFAULT_PACE_THRESHOLD = 2.0  # seconds; faster replies count as too fast
DEFAULT_PACE_STEP = 0.75  # extra seconds added for each consecutive fast reply
DEFAULT_PACE_MAX = 5.0  # longest pacing delay, in seconds
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 280
DEFAULT_LOG_DIR = "logs"


@dataclass
class BotConfig:
    name: str
    model: str
    replies: int
    system: str | None
    persona: str


@dataclass
class Config:
    left: BotConfig
    right: BotConfig
    topic: str
    starter: str
    pace_threshold: float
    pace_step: float
    pace_max: float
    temperature: float
    max_tokens: int
    base_url: str
    observer: str

    def bot(self, side: str) -> BotConfig:
        return self.left if side == "left" else self.right

    def summary(self) -> str:
        def cap(limit: int) -> str:
            return "unlimited" if limit < 0 else str(limit)

        return "\n".join(
            [
                f"left: {self.left.name} | {self.left.model} | replies {cap(self.left.replies)}",
                f"right: {self.right.name} | {self.right.model} | replies {cap(self.right.replies)}",
                f"observer: {self.observer}",
                f"starter: {self.starter}",
                f"topic: {self.topic}",
                f"pace threshold: {self.pace_threshold}s",
                f"pace step: {self.pace_step}s",
                f"pace max: {self.pace_max}s",
                f"temperature: {self.temperature}",
                f"max tokens: {self.max_tokens}",
                f"base url: {self.base_url}",
            ]
        )


@dataclass
class Turn:
    side: str
    name: str
    text: str


class Pacer:
    """Slow the next reply when a model answers faster than the threshold.

    The wait is the time still needed to reach the threshold, plus one extra
    step for every consecutive fast reply after the first. A reply that takes
    at least the threshold resets that streak.
    """

    def __init__(self, threshold: float, step: float, max_delay: float) -> None:
        self.threshold = threshold
        self.step = step
        self.max_delay = max_delay
        self.streak = 0

    def next_delay(self, elapsed: float) -> float:
        if self.threshold <= 0 or elapsed >= self.threshold:
            self.streak = 0
            return 0.0
        self.streak += 1
        base = self.threshold - elapsed
        extra = self.step * (self.streak - 1)
        return min(self.max_delay, max(0.0, base + extra))


class ChatLog:
    def __init__(self, path: Path, verbose: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.verbose = verbose
        self._file = path.open("a", encoding="utf-8")
        self._closed = False

    def record(self, label: str, text: str, *, status: bool = False) -> None:
        if self._closed or (status and not self.verbose):
            return
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._file.write(f"--- {stamp} {label}\n{text.rstrip()}\n\n")
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.close()


def other_side(side: str) -> str:
    return "right" if side == "left" else "left"


def reply_label(count: int, limit: int) -> str:
    cap = "∞" if limit < 0 else str(limit)
    return f"{count}/{cap}"


def can_reply(count: int, limit: int) -> bool:
    return limit < 0 or count < limit


def wrap_message(text: str, width: int) -> list[str]:
    width = max(1, width)
    lines: list[str] = []
    paragraphs = text.splitlines() or [""]
    for paragraph in paragraphs:
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


def panel_width(container_width: int, text: str, title: str) -> int:
    """Bubble width: shrink to the text, but wrap once a side's column is full."""
    if container_width < 12:
        return max(container_width, 1)
    column = min(container_width, max(28, int(container_width * 0.56)))
    inner_cap = max(8, column - 4)
    longest = len(title)
    for paragraph in text.splitlines() or [""]:
        longest = max(longest, len(paragraph))
    inner = min(inner_cap, max(longest, 1))
    return min(container_width, inner + 4)


def default_system(
    bot: BotConfig,
    other: BotConfig,
    topic: str,
    observer: str,
    *,
    include_observer: bool,
) -> str:
    prompt = (
        f"You are {bot.name}, talking with {other.name}. "
        f"The subject is: {topic}. {bot.persona} "
        "Write only your own next message, one to four sentences. "
        "Do not prefix your name, do not speak as the other participant, and do not narrate actions."
    )
    if include_observer:
        prompt += (
            f" {observer} has joined the conversation. Both of you can see their messages. "
            f"When {observer} speaks, respond to them."
        )
    return prompt


def messages_for(side: str, history: list[Turn], config: Config) -> list[dict[str, str]]:
    bot = config.bot(side)
    other = config.bot(other_side(side))
    observer_has_spoken = any(turn.side == "user" for turn in history)
    system = bot.system or default_system(
        bot,
        other,
        config.topic,
        config.observer,
        include_observer=observer_has_spoken,
    )
    payload: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history:
        if turn.side == side:
            payload.append({"role": "assistant", "content": turn.text})
        elif turn.side == "user":
            payload.append({"role": "user", "content": f"[{config.observer}]: {turn.text}"})
        else:
            payload.append({"role": "user", "content": f"[{turn.name}]: {turn.text}"})
    if len(payload) == 1:
        payload.append(
            {
                "role": "user",
                "content": (
                    f"Open the conversation about: {config.topic}. "
                    f"Speak only as {bot.name}."
                ),
            }
        )
    elif payload[-1]["role"] != "user":
        payload.append(
            {
                "role": "user",
                "content": (
                    f"Continue as {bot.name}. Reply to what was just said. "
                    "Write only your next message."
                ),
            }
        )
    return payload


def strip_name_prefix(name: str, text: str) -> str:
    stripped = text.strip()
    for prefix in (f"{name}:", f"{name}："):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def short_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > 240:
        return text[:237] + "..."
    return text or exc.__class__.__name__


def coerce_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "".join(parts)
    return str(content)


class OpenRouterCompleter:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90.0,
            default_headers={
                "HTTP-Referer": "https://localhost/chatbotv2",
                "X-Title": "chatbotv2",
            },
        )

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        on_delta: Callable[[str], None],
    ) -> str:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            piece = coerce_content(chunk.choices[0].delta.content)
            if not piece:
                continue
            parts.append(piece)
            on_delta(piece)
        return "".join(parts)


class Bubble(Widget):
    body = ""

    def __init__(self, name: str, side: str, text: str, stamp: str = "") -> None:
        super().__init__()
        self.speaker = name
        self.side = side
        self.body = text
        self.stamp = stamp or datetime.now().strftime("%H:%M")

    def set_body(self, text: str) -> None:
        self.body = text
        self.refresh(layout=True)

    def _lines(self, width: int) -> list[str]:
        text = self.body or "…"
        if width < 8:
            return wrap_message(text, max(1, width))
        outer = min(width, panel_width(width, text, self.speaker))
        inner = max(1, outer - 4)
        content = wrap_message(text, inner)
        bar = outer - 2
        title = f" {self.speaker} "
        stamp = f" {self.stamp} " if self.stamp else ""
        if len(title) + len(stamp) > bar:
            stamp = ""
        if len(title) > bar:
            title = title[:bar]
        gap = max(0, bar - len(title) - len(stamp))
        top = "╭" + title + ("─" * gap) + stamp + "╮"
        middle = [f"│ {line.ljust(inner)} │" for line in content]
        bottom = "╰" + ("─" * bar) + "╯"
        block = [line[:outer].ljust(outer) for line in (top, *middle, bottom)]
        if self.side == "right":
            return [line.rjust(width) for line in block]
        if self.side == "user":
            return [line.center(width) for line in block]
        return block

    def _visual(self, width: int) -> Text:
        style = {
            "left": "bold #7ec8ff",
            "right": "bold #8ee0b2",
            "user": "bold #f0c674",
            "notice": "#8b97a8",
        }.get(self.side, "#8b97a8")
        text = Text()
        for index, line in enumerate(self._lines(width)):
            if index:
                text.append("\n")
            text.append(line, style=style)
        return text

    def render(self) -> RenderableType:
        return self._visual(self.size.width or 1)

    def get_content_height(self, container, viewport, width: int) -> int:
        if width <= 0:
            return 0
        visual = visualize(self, self._visual(width), markup=False)
        return visual.get_height(self.styles, width)


class HeaderBar(Static):
    def __init__(self, config: Config) -> None:
        super().__init__(id="header")
        self.config = config

    def render(self) -> RenderableType:
        width = self.size.width or 80
        left = f"{self.config.left.name}  {self.config.left.model}"
        right = f"{self.config.right.model}  {self.config.right.name}"
        if len(left) + len(right) + 2 > width:
            budget = max(4, (width - 2) // 2)
            left = _clip(left, budget)
            right = _clip(right, budget)
        gap = max(1, width - len(left) - len(right))
        return Text.assemble(
            (left, "bold #7ec8ff"),
            " " * gap,
            (right, "bold #8ee0b2"),
        )


class StatusBar(Static):
    _text = ""
    _kind = "info"

    def set_line(self, text: str, kind: str) -> None:
        self._text = text
        self._kind = kind
        self.refresh()

    def render(self) -> RenderableType:
        style = {
            "thinking": "#f0c674",
            "replying": "#8ee0b2",
            "waiting": "#7ec8ff",
            "error": "#ff7b72",
            "idle": "#9aa4b5",
            "info": "#c5ced9",
        }.get(self._kind, "#c5ced9")
        return Text(self._text, style=style)


def _clip(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


class Transcript(ScrollableContainer):
    async def add(self, bubble: Bubble) -> None:
        await self.mount(bubble)
        self.scroll_end(animate=False)

    def follow_bottom(self) -> None:
        self.scroll_end(animate=False)


class ChatApp(App[None]):
    TITLE = "chat"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen {
        layout: vertical;
        background: #12141a;
    }
    #header {
        height: 1;
        background: #1a2030;
        padding: 0 1;
    }
    #transcript {
        height: 1fr;
        background: #12141a;
        padding: 1 1 0 1;
    }
    Bubble {
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
    }
    #composer {
        height: 3;
        margin: 0 1;
        border: round #3c4d68;
        background: #181c26;
        color: #e7ecf3;
    }
    #composer:focus {
        border: round #79a8ff;
    }
    #status {
        height: 1;
        background: #10131a;
        padding: 0 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("pageup", "history_up", "Older", priority=True),
        Binding("pagedown", "history_down", "Newer", priority=True),
    ]

    def __init__(self, config: Config, completer, chat_log: ChatLog) -> None:
        super().__init__()
        self.config = config
        self.completer = completer
        self.chat_log = chat_log
        self.history: list[Turn] = []
        self.counts = {"left": 0, "right": 0}
        self.turn = config.starter
        self.pacer = Pacer(config.pace_threshold, config.pace_step, config.pace_max)
        self.user_version = 0
        self._cancel_reply = asyncio.Event()
        self._stop = asyncio.Event()
        self._finished = False
        self._worker = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(self.config)
        yield Transcript(id="transcript")
        yield Input(
            placeholder=f"Message as {self.config.observer} — Enter sends, /quit exits",
            id="composer",
        )
        yield StatusBar(id="status")

    def on_mount(self) -> None:
        self.query_one("#composer", Input).focus()
        self._set_status("starting", "info")
        self._worker = self.run_chat()

    def on_unmount(self) -> None:
        self._stop.set()
        self._cancel_reply.set()
        self.chat_log.record("status", "session ended", status=True)
        self.chat_log.close()

    def action_history_up(self) -> None:
        self.query_one("#transcript", Transcript).scroll_page_up(animate=False)

    def action_history_down(self) -> None:
        self.query_one("#transcript", Transcript).scroll_page_down(animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text.lower() in {"/quit", "/exit"}:
            self.exit()
            return
        name = self.config.observer
        self.history.append(Turn("user", name, text))
        self.user_version += 1
        self._cancel_reply.set()
        self.chat_log.record(name, text)
        transcript = self.query_one("#transcript", Transcript)
        await transcript.add(Bubble(name, "user", text))
        if self._finished:
            self._set_status(
                "reply limit reached — message logged, the bots will not answer",
                "idle",
            )

    def _counts_label(self) -> str:
        left = self.config.left
        right = self.config.right
        return (
            f"{left.name} {reply_label(self.counts['left'], left.replies)}"
            f" · {right.name} {reply_label(self.counts['right'], right.replies)}"
        )

    def _set_status(self, text: str, kind: str) -> None:
        self.query_one("#status", StatusBar).set_line(
            f"{text}  ·  {self._counts_label()}",
            kind,
        )

    def _pick_speaker(self) -> str | None:
        if can_reply(self.counts[self.turn], self.config.bot(self.turn).replies):
            return self.turn
        other = other_side(self.turn)
        if can_reply(self.counts[other], self.config.bot(other).replies):
            self.turn = other
            return other
        return None

    @work(exclusive=True, exit_on_error=False)
    async def run_chat(self) -> None:
        transcript = self.query_one("#transcript", Transcript)
        try:
            await transcript.add(
                Bubble(
                    "session",
                    "notice",
                    f"Logging to {self.chat_log.path}. "
                    "Type a message and press Enter to interrupt. /quit exits.",
                )
            )
            while not self._stop.is_set():
                side = self._pick_speaker()
                if side is None:
                    self._finished = True
                    self._set_status("reply limit reached", "idle")
                    self.chat_log.record("status", "reply limit reached", status=True)
                    return
                produced = await self._speak(side, transcript)
                if produced is None:
                    return
                if not produced:
                    continue
                self.turn = other_side(side)
                bot = self.config.bot(self.turn)
                if not can_reply(self.counts[self.turn], bot.replies):
                    continue
                await self._hold(bot.name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = short_error(exc)
            self._finished = True
            self.chat_log.record("error", message)
            with contextlib.suppress(Exception):
                self._set_status(f"stopped · {message}", "error")

    async def _speak(self, side: str, transcript: Transcript) -> bool | None:
        """Return True when a reply was kept, False when interrupted, None on failure."""
        bot = self.config.bot(side)
        version = self.user_version
        self._cancel_reply.clear()
        if self.user_version != version or self._stop.is_set():
            return False

        self._set_status(f"thinking · {bot.name} · {bot.model}", "thinking")
        bubble = Bubble(bot.name, side, "…")
        await transcript.add(bubble)
        history_at_start = len(self.history)
        collected: list[str] = []
        started = False

        def on_delta(piece: str) -> None:
            nonlocal started
            if not started:
                collected.clear()
                started = True
                self._set_status(f"replying · {bot.name} · {bot.model}", "replying")
            collected.append(piece)
            bubble.set_body("".join(collected))
            transcript.follow_bottom()

        started_at = time.monotonic()
        task = asyncio.create_task(
            self.completer.complete(
                model=bot.model,
                messages=messages_for(side, list(self.history), self.config),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                on_delta=on_delta,
            )
        )
        waiter = asyncio.create_task(self._cancel_reply.wait())
        done, _pending = await asyncio.wait(
            {task, waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task not in done:
            task.cancel()
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await waiter
            await bubble.remove()
            self.chat_log.record(
                "status",
                f"interrupted {bot.name} to read {self.config.observer}",
                status=True,
            )
            return False

        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter

        try:
            raw = task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await bubble.remove()
            message = short_error(exc)
            self._finished = True
            self._set_status(f"stopped · {message}", "error")
            self.chat_log.record("error", message)
            await transcript.add(Bubble("error", "notice", message))
            return None

        text = strip_name_prefix(bot.name, raw)
        if not text:
            text = "(no reply)"
        bubble.set_body(text)
        elapsed = time.monotonic() - started_at
        turn = Turn(side, bot.name, text)
        arrived = self.history[history_at_start:]
        self.history[history_at_start:] = [turn, *arrived]
        self.counts[side] += 1
        self.chat_log.record(bot.name, text)
        self._last_elapsed = elapsed
        return True

    async def _hold(self, next_name: str) -> None:
        elapsed = getattr(self, "_last_elapsed", 0.0)
        delay = self.pacer.next_delay(elapsed)
        if delay <= 0 or self._stop.is_set():
            return
        self.chat_log.record(
            "pace",
            (
                f"holding {delay:.2f}s before {next_name} "
                f"(reply took {elapsed:.2f}s, fast streak {self.pacer.streak})"
            ),
            status=True,
        )
        version = self.user_version
        started = time.monotonic()
        while not self._stop.is_set():
            remaining = delay - (time.monotonic() - started)
            if remaining <= 0:
                return
            if self.user_version != version:
                self._set_status(f"{self.config.observer} joined — answering now", "info")
                self.chat_log.record(
                    "pace",
                    f"wait skipped because {self.config.observer} spoke",
                    status=True,
                )
                return
            self._set_status(
                f"waiting {remaining:.1f}s · next {next_name} · fast ×{self.pacer.streak}",
                "waiting",
            )
            await asyncio.sleep(min(0.2, remaining))

def load_api_key() -> str:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
    for name in ("OPENROUTER_API", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip().strip('"').strip("'")
        if value:
            return value
    raise SystemExit(
        "No API key found. Put OPENROUTER_API in .env "
        "(OpenRouter keys also work as OPENROUTER_API_KEY, OpenAI keys as OPENAI_API_KEY)."
    )


def resolve_log_path(log: str | None, log_dir: str) -> Path:
    stamp = datetime.now().strftime("chat-%Y%m%d-%H%M%S.log")
    if not log:
        return Path(log_dir) / stamp
    path = Path(log).expanduser()
    if log.endswith(("/", os.sep)) or (path.exists() and path.is_dir()):
        return path / stamp
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chat.py",
        description=(
            "A console chat between two models. Each bot keeps to one side of the "
            "screen. Messages you type are shown to both, and either bot can answer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            pacing:
              If a reply is produced faster than --pace-threshold seconds, the next
              reply waits. The wait is (threshold - elapsed) plus --pace-step for
              every further fast reply in a row, capped by --pace-max. A slower
              reply resets the streak. A message from you skips whatever wait is left.

            examples:
              .venv/bin/python chat.py
              .venv/bin/python chat.py --replies -1 --topic "train schedules"
              .venv/bin/python chat.py --left-model openai/gpt-4o-mini \\
                  --right-model anthropic/claude-3.5-haiku --left-replies 6 --right-replies -1
              .venv/bin/python chat.py --pace-threshold 3 --pace-step 1 --pace-max 12 \\
                  --log logs/session.log
            """
        ),
    )
    parser.add_argument("--left-model", default=DEFAULT_LEFT_MODEL, help=f"OpenRouter model id for the left bot (default: {DEFAULT_LEFT_MODEL})")
    parser.add_argument("--right-model", default=DEFAULT_RIGHT_MODEL, help=f"OpenRouter model id for the right bot (default: {DEFAULT_RIGHT_MODEL})")
    parser.add_argument("--left-name", default=DEFAULT_LEFT_NAME, help=f"display name for the left bot (default: {DEFAULT_LEFT_NAME})")
    parser.add_argument("--right-name", default=DEFAULT_RIGHT_NAME, help=f"display name for the right bot (default: {DEFAULT_RIGHT_NAME})")
    parser.add_argument("--observer-name", default=DEFAULT_OBSERVER_NAME, help=f"display name for you, the observer (default: {DEFAULT_OBSERVER_NAME})")
    parser.add_argument(
        "--replies",
        type=int,
        default=DEFAULT_REPLIES,
        help=f"reply limit for each bot, unless overridden; -1 is unlimited (default: {DEFAULT_REPLIES})",
    )
    parser.add_argument("--left-replies", type=int, default=None, help="reply limit for the left bot; -1 is unlimited")
    parser.add_argument("--right-replies", type=int, default=None, help="reply limit for the right bot; -1 is unlimited")
    parser.add_argument("--starter", choices=("left", "right"), default=DEFAULT_STARTER, help=f"which bot speaks first (default: {DEFAULT_STARTER})")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="what the bots talk about")
    parser.add_argument("--left-system", default=None, help="replace the left bot's system prompt")
    parser.add_argument("--right-system", default=None, help="replace the right bot's system prompt")
    parser.add_argument(
        "--pace-threshold",
        type=float,
        default=DEFAULT_PACE_THRESHOLD,
        help=f"replies faster than this many seconds count as too fast (default: {DEFAULT_PACE_THRESHOLD})",
    )
    parser.add_argument(
        "--pace-step",
        type=float,
        default=DEFAULT_PACE_STEP,
        help=f"extra delay added for each consecutive fast reply (default: {DEFAULT_PACE_STEP})",
    )
    parser.add_argument(
        "--pace-max",
        type=float,
        default=DEFAULT_PACE_MAX,
        help=f"longest pacing delay, in seconds (default: {DEFAULT_PACE_MAX})",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"token cap for each reply (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument(
        "--log",
        default=None,
        help="log file path; a directory (or a trailing slash) gets a timestamped file inside it",
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help=f"directory for timestamped logs when --log is omitted (default: {DEFAULT_LOG_DIR})")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also log pace and status lines; without this, the log is the conversation",
    )
    args = parser.parse_args(argv)
    _require_limit(parser, "--replies", args.replies)
    if args.left_replies is not None:
        _require_limit(parser, "--left-replies", args.left_replies)
    if args.right_replies is not None:
        _require_limit(parser, "--right-replies", args.right_replies)
    for flag, value in (
        ("--pace-threshold", args.pace_threshold),
        ("--pace-step", args.pace_step),
        ("--pace-max", args.pace_max),
    ):
        if value < 0:
            parser.error(f"{flag} must be zero or greater")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if not args.left_name.strip() or not args.right_name.strip() or not args.observer_name.strip():
        parser.error("bot and observer names cannot be empty")
    if not args.left_model.strip() or not args.right_model.strip():
        parser.error("model names cannot be empty")
    return args


def _require_limit(parser: argparse.ArgumentParser, flag: str, value: int) -> None:
    if value < -1:
        parser.error(f"{flag} must be -1 (unlimited) or zero or greater")


def config_from_args(args: argparse.Namespace) -> Config:
    left_replies = args.replies if args.left_replies is None else args.left_replies
    right_replies = args.replies if args.right_replies is None else args.right_replies
    return Config(
        left=BotConfig(
            name=args.left_name.strip(),
            model=args.left_model.strip(),
            replies=left_replies,
            system=args.left_system,
            persona=LEFT_PERSONA,
        ),
        right=BotConfig(
            name=args.right_name.strip(),
            model=args.right_model.strip(),
            replies=right_replies,
            system=args.right_system,
            persona=RIGHT_PERSONA,
        ),
        topic=args.topic,
        starter=args.starter,
        pace_threshold=args.pace_threshold,
        pace_step=args.pace_step,
        pace_max=args.pace_max,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        observer=args.observer_name.strip(),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    api_key = load_api_key()
    log_path = resolve_log_path(args.log, args.log_dir)
    chat_log = ChatLog(log_path, verbose=args.verbose)
    chat_log.record("config", config.summary())
    app = ChatApp(config, OpenRouterCompleter(api_key, config.base_url), chat_log)
    try:
        app.run()
    finally:
        chat_log.close()


if __name__ == "__main__":
    main()
