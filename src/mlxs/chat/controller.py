"""Chat controller — UI-agnostic conversation orchestrator.

Manages session state, prompt building, generation, prompt cache integration,
and output sanitisation.  Produces events via callbacks so any UI (TUI, simple
CLI, future interfaces) can consume them without coupling.

No print(), no Textual imports, no HTTP concepts.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from mlxs._types import FinishReason, GenerateOptions
from mlxs.chat.session import ChatSession
from mlxs.chat.store import SessionStore
from mlxs.chat.template import (
    build_prompt_ids,
    collect_stop_token_ids,
    sanitize_assistant_text,
)
from mlxs.server.deps import Dependencies

logger = logging.getLogger(__name__)


# -- Events ----------------------------------------------------------------

class EventKind(Enum):
    GENERATION_STARTED = auto()
    TOKEN_CHUNK = auto()
    GENERATION_FINISHED = auto()
    ERROR = auto()
    STATUS = auto()


@dataclass(slots=True)
class ChatEvent:
    kind: EventKind
    text: str = ""
    finish_reason: FinishReason | None = None
    token_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for event callbacks
EventCallback = Callable[[ChatEvent], None]


class ChatController:
    """Orchestrates multi-turn chat: session, template, generate, cache."""

    def __init__(
        self,
        deps: Dependencies,
        store: SessionStore | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._deps = deps
        self._store = store or SessionStore()
        self._on_event = on_event or (lambda _: None)
        self._session = ChatSession(model_path=deps.config.model.model_path or "")
        self._generating = False
        self._cancel = threading.Event()

        # Discover extra stop tokens once
        self._extra_stop_ids = collect_stop_token_ids(deps.tokenizer)
        self._gen_opts = self._build_gen_opts()

    # -- Properties --------------------------------------------------------

    @property
    def session(self) -> ChatSession:
        return self._session

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def model_path(self) -> str:
        return self._deps.config.model.model_path or ""

    @property
    def temperature(self) -> float:
        return self._gen_opts.temperature

    @property
    def max_tokens(self) -> int:
        return self._gen_opts.max_tokens

    # -- Core chat flow ----------------------------------------------------

    def send_message(self, content: str) -> None:
        """Send a user message and generate assistant response.

        This is blocking — call from a worker thread in TUI context.
        Events are emitted via on_event callback.
        """
        if self._generating:
            self._emit(ChatEvent(kind=EventKind.ERROR, text="Generation already in progress"))
            return

        content = content.strip()
        if not content:
            return

        self._cancel.clear()
        self._session.add_user_message(content)
        self._session.auto_title()

        try:
            prompt_token_ids = build_prompt_ids(
                self._deps.tokenizer,
                self._session.prompt_messages(),
            )
        except Exception as exc:
            self._session.messages.pop()
            self._emit(ChatEvent(kind=EventKind.ERROR, text=f"Template error: {exc}"))
            return

        if not prompt_token_ids:
            self._session.messages.pop()
            self._emit(ChatEvent(kind=EventKind.ERROR, text="Empty prompt"))
            return

        self._generating = True
        self._emit(ChatEvent(kind=EventKind.GENERATION_STARTED))

        model_id = self._deps.config.model.model_path or "default"
        cache_state, prefix_len = self._deps.prompt_cache.get(
            model_id, tuple(prompt_token_ids),
        )
        suffix_len = len(prompt_token_ids) - prefix_len

        if cache_state is not None and prefix_len > 0 and suffix_len > 0:
            prompt_for_gen: list[int] = list(prompt_token_ids[prefix_len:])
            cache_for_gen = cache_state
        else:
            prompt_for_gen = prompt_token_ids
            cache_for_gen = None

        final_cache_ref: list[Any] = []
        gen_kwargs: dict[str, Any] = {
            "prefill_step_size": self._deps.config.generate.prefill_step_size,
            "compile_decode": self._deps.config.generate.compile_decode,
            "clear_cache_interval": self._deps.config.generate.clear_cache_interval,
            "final_cache_out": final_cache_ref,
        }
        if cache_for_gen is not None:
            gen_kwargs["cache"] = cache_for_gen

        generated_ids: list[int] = []
        collected_text: list[str] = []

        try:
            for event in self._deps.generate_fn(
                self._deps.model,
                self._deps.tokenizer,
                prompt_for_gen,
                self._gen_opts,
                **gen_kwargs,
            ):
                if self._cancel.is_set():
                    break

                generated_ids.append(event.token_id)
                collected_text.append(event.text)
                self._emit(ChatEvent(
                    kind=EventKind.TOKEN_CHUNK,
                    text=event.text,
                    token_id=event.token_id,
                ))

                if event.finish_reason is not None:
                    break

        except Exception as exc:
            self._generating = False
            self._session.messages.pop()  # remove user message
            self._emit(ChatEvent(kind=EventKind.ERROR, text=f"Generation error: {exc}"))
            return

        # Update prompt cache
        cache_to_put = (
            cache_for_gen
            if cache_for_gen is not None
            else (final_cache_ref[0] if final_cache_ref else None)
        )
        if cache_to_put is not None:
            new_prefix = tuple(prompt_token_ids) + tuple(generated_ids)
            self._deps.prompt_cache.put(model_id, new_prefix, cache_to_put)

        # Build and sanitise full response
        if generated_ids:
            full_text = sanitize_assistant_text(
                self._deps.tokenizer.decode(generated_ids),
            )
        else:
            full_text = ""

        if self._cancel.is_set():
            # Cancelled: still save partial if we have text
            if full_text:
                self._session.add_assistant_message(
                    full_text, metadata={"finish_reason": "cancelled"},
                )
        else:
            self._session.add_assistant_message(full_text)

        self._generating = False
        self._emit(ChatEvent(
            kind=EventKind.GENERATION_FINISHED,
            text=full_text,
        ))

    def cancel_generation(self) -> None:
        """Request cancellation of the current generation."""
        self._cancel.set()

    # -- Session management ------------------------------------------------

    def new_session(self) -> ChatSession:
        """Create a new empty session, replacing the current one."""
        self.save_session()
        self._session = ChatSession(
            model_path=self._deps.config.model.model_path or "",
        )
        self._emit(ChatEvent(kind=EventKind.STATUS, text="New session created"))
        return self._session

    def save_session(self) -> None:
        """Persist current session."""
        if self._session.messages:
            self._store.save(self._session)
            self._emit(ChatEvent(kind=EventKind.STATUS, text="Session saved"))

    def load_session(self, session_id: str) -> ChatSession:
        """Load a session by id, replacing the current one."""
        self.save_session()
        self._session = self._store.load(session_id)
        self._emit(ChatEvent(kind=EventKind.STATUS, text=f"Loaded session: {self._session.title}"))
        return self._session

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._store.list_sessions()

    def rename_session(self, new_title: str) -> None:
        self._session.title = new_title
        self._store.save(self._session)
        self._emit(ChatEvent(kind=EventKind.STATUS, text=f"Renamed to: {new_title}"))

    def delete_session(self, session_id: str) -> bool:
        return self._store.delete(session_id)

    # -- Editing -----------------------------------------------------------

    def retry_last(self) -> None:
        """Remove the last assistant reply and re-generate."""
        if self._generating:
            self._emit(ChatEvent(kind=EventKind.ERROR, text="Cannot retry while generating"))
            return
        if not self._session.messages:
            return
        # Remove last assistant message
        if self._session.messages[-1].role == "assistant":
            self._session.messages.pop()
        # Re-send the last user message
        if self._session.messages and self._session.messages[-1].role == "user":
            last_user = self._session.messages.pop()
            self.send_message(last_user.content)

    def delete_last_turn(self) -> int:
        """Remove last user+assistant exchange."""
        removed = self._session.delete_last_turn()
        if removed:
            self._emit(ChatEvent(kind=EventKind.STATUS, text=f"Deleted {removed} message(s)"))
        return removed

    def clear_session(self) -> None:
        """Clear all messages in current session."""
        self._session.clear_messages()
        self._emit(ChatEvent(kind=EventKind.STATUS, text="Session cleared"))

    # -- Internal ----------------------------------------------------------

    def _build_gen_opts(self) -> GenerateOptions:
        c = self._deps.config.generate
        merged_eos = tuple(sorted(set(c.extra_eos_token_ids) | set(self._extra_stop_ids)))
        return GenerateOptions(
            max_tokens=c.max_tokens,
            temperature=c.temperature,
            top_p=c.top_p,
            top_k=c.top_k,
            min_p=c.min_p,
            seed=c.seed,
            stop_sequences=c.stop_sequences,
            extra_eos_token_ids=merged_eos,
            repetition_penalty=c.repetition_penalty,
            logprobs=c.logprobs,
            top_logprobs=c.top_logprobs,
            stream=True,
        )

    def _emit(self, event: ChatEvent) -> None:
        try:
            self._on_event(event)
        except Exception:
            logger.exception("Error in event callback")
