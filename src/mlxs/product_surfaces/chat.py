"""Canonical Layer 4 CLI chat surface."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlxs.advanced_engines.prompt_cache import PromptCachePlan
from mlxs._types import FinishReason, GenerateOptions
from mlxs.chat.input import AttachmentResolutionError, resolve_attachments
from mlxs.chat.session import ChatSession
from mlxs.chat.template import (
    StreamingTextSanitizer,
    build_prompt_ids,
    collect_stop_token_ids,
    sanitize_assistant_text,
)
from mlxs.config.schema import GenerateConfig

if TYPE_CHECKING:
    from mlxs.product_surfaces.bootstrap import ProductRuntime

logger = logging.getLogger(__name__)


def parse_command(line: str) -> tuple[str, str] | None:
    """Parse a slash command for the Layer 4 CLI chat surface."""
    if not line.startswith("/"):
        return None
    body = line[1:].strip()
    if not body:
        return "", ""
    name, _, rest = body.partition(" ")
    return name.lower(), rest.strip()


def help_card() -> str:
    """Return a lightweight help card without importing prompt-toolkit UI code."""
    return "\n".join(
        (
            "/help",
            "/quit",
            "/new",
            "/clear",
            "/undo",
            "/retry",
            "/system <text>",
            "/title <text>",
            "/history [n]",
            "/export [path]",
            "/stats",
            "/status",
        )
    )


def run_chat_loop(
    deps: ProductRuntime,
    options: GenerateOptions | None = None,
    initial_query: str | None = None,
) -> None:
    """Run the docked shell or the plain streaming chat path."""
    extra_stop_ids = collect_stop_token_ids(deps.tokenizer)
    gen_opts = (
        options
        if options is not None
        else _options_from_config(
            deps.config.generate,
            extra_stop_ids=extra_stop_ids,
        )
    )
    model_id = _model_id_for_cache(deps)

    logger.info("Chat started (model_id=%s).", model_id)

    if initial_query is None and _supports_interactive_shell():
        _InteractiveChatController(deps, gen_opts, model_id).run()
        logger.info("Chat exiting.")
        return

    _run_plain_chat_loop(deps, gen_opts, model_id, initial_query=initial_query)
    logger.info("Chat exiting.")


class _InteractiveChatController:
    """Owns interactive shell state and generation worker lifecycle."""

    def __init__(self, deps: ProductRuntime, gen_opts: GenerateOptions, model_id: str) -> None:
        from mlxs.chat.cli import ChatShell

        self._deps = deps
        self._gen_opts = gen_opts
        self._model_id = model_id
        self._session = ChatSession(model_path=model_id)
        self._shell = ChatShell(
            model_id,
            self._session,
            max_tokens=gen_opts.max_tokens,
            temperature=gen_opts.temperature,
        )
        self._cancel_event = threading.Event()
        self._worker: threading.Thread | None = None

    def run(self) -> None:
        self._shell.run(on_submit=self._submit, on_cancel=self._cancel_generation)

    def _submit(self, raw_text: str) -> None:
        text = raw_text.strip()
        if not text:
            return
        if text == "?":
            self._shell.show_help()
            return
        if self._is_generating():
            self._shell.show_status("Generation already in progress. Press Esc to cancel it.")
            return
        if text.lower() == "/quit":
            self._shell.request_exit()
            return

        command = parse_command(text)
        if command is not None:
            self._handle_command(command)
            return

        try:
            attachments = resolve_attachments(text, cwd=Path.cwd())
        except AttachmentResolutionError as exc:
            self._shell.set_state("error", "Attachment error")
            self._shell.show_error(str(exc))
            return

        metadata = (
            {"attachments": [attachment.to_metadata() for attachment in attachments]}
            if attachments
            else None
        )
        self._session.add_user_message(text, metadata=metadata)
        self._session.auto_title()
        self._shell.sync_session(self._session)
        self._start_generation()

    def _handle_command(self, command: tuple[str, str]) -> None:
        name, arg = command
        if not name:
            self._shell.show_error("Empty command. Use /help.")
            return
        if name in {"quit", "exit"}:
            self._shell.request_exit()
            return
        if name == "help":
            self._shell.show_help()
            return
        if name == "model":
            self._shell.show_runtime(
                max_tokens=self._gen_opts.max_tokens,
                temperature=self._gen_opts.temperature,
            )
            return
        if name == "new":
            self._session = ChatSession(model_path=self._model_id)
            self._shell.sync_session(self._session, clear_notices=True)
            self._shell.show_status(f"Started a new chat: {self._session.session_id}")
            return
        if name == "clear":
            system_prompt = self._session.system_message()
            self._session.clear_messages()
            if system_prompt:
                self._session.set_system_message(system_prompt)
            self._shell.sync_session(self._session, clear_notices=True)
            self._shell.show_status("Conversation cleared.")
            return
        if name == "undo":
            removed = self._session.delete_last_turn()
            self._shell.sync_session(self._session)
            if removed:
                self._shell.show_status(f"Removed {removed} message(s).")
            else:
                self._shell.show_status("Nothing to undo.")
            return
        if name == "retry":
            self._retry_async()
            return
        if name == "system":
            self._session = _handle_system_command(self._session, arg, self._shell)
            self._shell.sync_session(self._session)
            return
        if name == "title":
            title = arg.strip()
            if not title:
                self._shell.show_error("Usage: /title <text>")
                return
            self._session.title = title
            self._shell.sync_session(self._session)
            self._shell.show_status(f"Renamed chat to: {title}")
            return
        if name == "history":
            limit = _parse_history_limit(arg, self._shell)
            if limit is not None:
                self._shell.show_history(self._session, limit)
            return
        if name == "export":
            _export_session(self._session, arg, self._shell)
            return
        if name in {"stats", "status"}:
            self._shell.show_stats(self._session)
            return
        self._shell.show_error(f"Unknown command: /{name}. Use /help.")

    def _start_generation(self) -> None:
        self._cancel_event = threading.Event()
        self._shell.set_state("generating", "Streaming reply")
        self._worker = threading.Thread(target=self._run_generation, daemon=True)
        self._worker.start()

    def _run_generation(self) -> None:
        status = _run_turn(
            self._deps,
            self._session,
            self._gen_opts,
            self._model_id,
            self._shell,
            cancel_event=self._cancel_event,
        )
        if status == "ok":
            self._shell.sync_session(self._session)
            self._shell.set_state("idle", "Ready")
            return
        if status == "cancel":
            self._shell.sync_session(self._session)
            self._shell.set_state("idle", "Cancelled")
            self._shell.show_status("Generation cancelled.")
            return
        if status == "interrupt":
            self._shell.sync_session(self._session)
            self._shell.set_state("idle", "Interrupted")
            self._shell.show_status("Generation interrupted.")
            return
        self._shell.sync_session(self._session)
        self._shell.set_state("error", "Last turn failed")

    def _cancel_generation(self) -> None:
        if not self._is_generating():
            return
        if not self._cancel_event.is_set():
            self._cancel_event.set()
            self._shell.set_state("cancelling", "Cancelling current turn")

    def _retry_async(self) -> None:
        if not self._session.messages:
            self._shell.show_status("Nothing to retry.")
            return
        if self._session.messages[-1].role == "assistant":
            self._session.messages.pop()
        if not self._session.messages or self._session.messages[-1].role != "user":
            self._shell.show_status("Nothing to retry.")
            return
        last_user = self._session.messages.pop()
        self._session.add_user_message(last_user.content, metadata=dict(last_user.metadata))
        self._session.auto_title()
        self._shell.sync_session(self._session)
        self._start_generation()

    def _is_generating(self) -> bool:
        return self._worker is not None and self._worker.is_alive()


class _PlainConsole:
    """Minimal renderer for one-shot and non-TTY chat flows."""

    __slots__ = ("_assistant_open",)

    def __init__(self) -> None:
        self._assistant_open = False

    def prompt(self) -> str | None:
        return "You> " if sys.stdin.isatty() and sys.stdout.isatty() else None

    def show_help(self) -> None:
        self._break_reply()
        print("Commands")
        print(help_card())

    def show_status(self, text: str) -> None:
        self._break_reply()
        print(f"// {text}")

    def show_error(self, text: str) -> None:
        self._break_reply()
        print(f"!! {text}")

    def show_stats(self, session: ChatSession) -> None:
        self._break_reply()
        turns = sum(1 for msg in session.messages if msg.role == "user")
        assistant_messages = sum(1 for msg in session.messages if msg.role == "assistant")
        system_prompt = session.system_message()
        print("Session")
        print(f"  title     {session.title}")
        print(f"  session   {session.session_id}")
        print(f"  model     {session.model_path or 'default'}")
        print(f"  user      {turns}")
        print(f"  assistant {assistant_messages}")
        print(f"  system    {'set' if system_prompt else 'unset'}")
        if system_prompt:
            print(f"  prompt    {_truncate(system_prompt, 88)}")

    def show_runtime(self, *, max_tokens: int, temperature: float) -> None:
        self._break_reply()
        print("Runtime")
        print(f"  max_tokens  {max_tokens}")
        print(f"  temperature {temperature}")

    def show_history(self, session: ChatSession, limit: int) -> None:
        self._break_reply()
        transcript = [msg for msg in session.messages if msg.role != "system"]
        if not transcript:
            print("No chat history yet.")
            return
        print("History")
        for message in transcript[-limit:]:
            role = {
                "assistant": "MLXs",
                "user": "You",
            }.get(message.role, message.role.capitalize())
            print(f"  {role:<6} {_truncate(message.content, 96)}")

    def stream_reply(self, text: str) -> None:
        if not text:
            return
        self._assistant_open = True
        print(text, end="", flush=True)

    def finish_reply(self, *, emitted_text: bool) -> None:
        if self._assistant_open or emitted_text:
            print()
        self._assistant_open = False

    def close_reply(self) -> None:
        self._break_reply()

    def _break_reply(self) -> None:
        if self._assistant_open:
            print()
            self._assistant_open = False


def _run_plain_chat_loop(
    deps: ProductRuntime,
    gen_opts: GenerateOptions,
    model_id: str,
    *,
    initial_query: str | None,
) -> None:
    session = ChatSession(model_path=model_id)
    console = _PlainConsole()

    if initial_query is not None:
        session.add_user_message(initial_query)
        session.auto_title()
        _run_turn(deps, session, gen_opts, model_id, console)
        return

    while True:
        try:
            line = _read_line(console.prompt())
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            break
        if line is None:
            break

        line = line.strip()
        if not line or line.lower() == "/quit":
            break
        if line == "?":
            console.show_help()
            continue

        command = parse_command(line)
        if command is not None:
            session, should_exit = _handle_plain_command(
                command,
                deps,
                session,
                gen_opts,
                model_id,
                console,
            )
            if should_exit:
                break
            continue

        try:
            attachments = resolve_attachments(line, cwd=Path.cwd())
        except AttachmentResolutionError as exc:
            console.show_error(str(exc))
            continue
        metadata = (
            {"attachments": [attachment.to_metadata() for attachment in attachments]}
            if attachments
            else None
        )
        session.add_user_message(line, metadata=metadata)
        session.auto_title()
        status = _run_turn(deps, session, gen_opts, model_id, console)
        if status == "interrupt":
            break


def _run_turn(
    deps: ProductRuntime,
    session: ChatSession,
    gen_opts: GenerateOptions,
    model_id: str,
    console: Any,
    *,
    cancel_event: threading.Event | None = None,
) -> str:
    """Generate one assistant turn from the current message list."""
    try:
        prompt_token_ids = build_prompt_ids(deps.tokenizer, session.prompt_messages())
    except Exception as exc:
        print(f"Error applying chat template: {exc}", file=sys.stderr)
        _drop_pending_user_message(session)
        return "retry"

    if not prompt_token_ids:
        _drop_pending_user_message(session)
        return "retry"

    cache_plan = deps.prompt_cache_orchestrator.prepare(model_id, prompt_token_ids)
    prompt_for_gen = cache_plan.prompt_for_generation
    cache_for_gen = cache_plan.cache_for_generation

    final_cache_ref: list[Any] = []
    gen_kwargs: dict[str, Any] = {
        "prefill_step_size": deps.config.generate.prefill_step_size,
        "compile_decode": deps.config.generate.compile_decode,
        "clear_cache_interval": deps.config.generate.clear_cache_interval,
        "final_cache_out": final_cache_ref,
    }
    if cache_for_gen is not None:
        gen_kwargs["cache"] = cache_for_gen

    generated_ids: list[int] = []
    sanitizer = StreamingTextSanitizer()
    emitted_text = False
    finish_reason: FinishReason | None = None
    try:
        for event in deps.generate_fn(
            deps.model,
            deps.tokenizer,
            prompt_for_gen,
            gen_opts,
            **gen_kwargs,
        ):
            if cancel_event is not None and cancel_event.is_set():
                console.close_reply()
                return "cancel"
            clean_chunk = sanitizer.feed(event.text)
            if clean_chunk:
                console.stream_reply(clean_chunk)
                emitted_text = True
            generated_ids.append(event.token_id)
            finish_reason = event.finish_reason
            if event.finish_reason is not None:
                break
    except KeyboardInterrupt:
        console.close_reply()
        print("Interrupted.", file=sys.stderr)
        _drop_pending_user_message(session)
        return "interrupt"
    except Exception as exc:
        console.close_reply()
        print(f"Generation error: {exc}", file=sys.stderr)
        _drop_pending_user_message(session)
        return "retry"

    if cancel_event is not None and cancel_event.is_set():
        console.close_reply()
        return "cancel"

    tail = sanitizer.flush()
    if tail:
        console.stream_reply(tail)
        emitted_text = True
    console.finish_reply(emitted_text=emitted_text)

    full_response = sanitize_assistant_text(deps.tokenizer.decode(generated_ids))
    metadata = {"finish_reason": finish_reason.name.lower()} if finish_reason is not None else None
    session.add_assistant_message(full_response, metadata=metadata)

    try:
        committed_prompt_ids = tuple(
            build_prompt_ids(
                deps.tokenizer,
                session.prompt_messages(),
                add_generation_prompt=False,
            )
        )
    except Exception:
        committed_prompt_ids = cache_plan.full_prompt_token_ids + tuple(generated_ids)

    deps.prompt_cache_orchestrator.commit(
        PromptCachePlan(
            model_id=cache_plan.model_id,
            full_prompt_token_ids=committed_prompt_ids,
            prompt_for_generation=cache_plan.prompt_for_generation,
            cache_for_generation=cache_plan.cache_for_generation,
            prefix_length=cache_plan.prefix_length,
        ),
        generated_ids=[],
        final_cache_out=final_cache_ref,
    )
    return "ok"


def _options_from_config(
    c: GenerateConfig,
    *,
    extra_stop_ids: tuple[int, ...] = (),
) -> GenerateOptions:
    """Build GenerateOptions from GenerateConfig, merging discovered stop ids."""
    merged_eos = tuple(sorted(set(c.extra_eos_token_ids) | set(extra_stop_ids)))
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


def _model_id_for_cache(deps: ProductRuntime) -> str:
    """Stable model id for prompt cache keys."""
    return deps.config.model.model_path or "default"


def _handle_plain_command(
    command: tuple[str, str],
    deps: ProductRuntime,
    session: ChatSession,
    gen_opts: GenerateOptions,
    model_id: str,
    console: _PlainConsole,
) -> tuple[ChatSession, bool]:
    """Handle a slash command in the plain chat path."""
    name, arg = command
    if not name:
        console.show_error("Empty command. Use /help.")
        return session, False
    if name in {"quit", "exit"}:
        return session, True
    if name == "help":
        console.show_help()
        return session, False
    if name == "model":
        console.show_runtime(
            max_tokens=gen_opts.max_tokens,
            temperature=gen_opts.temperature,
        )
        return session, False
    if name == "new":
        new_session = ChatSession(model_path=model_id)
        console.show_status(f"Started a new chat: {new_session.session_id}")
        return new_session, False
    if name == "clear":
        system_prompt = session.system_message()
        session.clear_messages()
        if system_prompt:
            session.set_system_message(system_prompt)
        console.show_status("Conversation cleared.")
        return session, False
    if name == "undo":
        removed = session.delete_last_turn()
        if removed:
            console.show_status(f"Removed {removed} message(s).")
        else:
            console.show_status("Nothing to undo.")
        return session, False
    if name == "retry":
        status = _retry_last_turn(deps, session, gen_opts, model_id, console)
        return session, status == "interrupt"
    if name == "system":
        return _handle_system_command(session, arg, console), False
    if name == "title":
        title = arg.strip()
        if not title:
            console.show_error("Usage: /title <text>")
            return session, False
        session.title = title
        console.show_status(f"Renamed chat to: {title}")
        return session, False
    if name == "history":
        limit = _parse_history_limit(arg, console)
        if limit is not None:
            console.show_history(session, limit)
        return session, False
    if name == "export":
        _export_session(session, arg, console)
        return session, False
    if name in {"stats", "status"}:
        console.show_stats(session)
        return session, False
    console.show_error(f"Unknown command: /{name}. Use /help.")
    return session, False


def _handle_system_command(
    session: ChatSession,
    arg: str,
    console: Any,
) -> ChatSession:
    """Inspect or update the session system prompt."""
    text = arg.strip()
    if not text:
        current = session.system_message()
        if current:
            console.show_status(f"System prompt: {current}")
        else:
            console.show_status("System prompt is not set.")
        return session
    if text.lower() in {"clear", "off", "none"}:
        session.set_system_message("")
        console.show_status("System prompt cleared.")
        return session
    session.set_system_message(text)
    console.show_status("System prompt updated.")
    return session


def _retry_last_turn(
    deps: ProductRuntime,
    session: ChatSession,
    gen_opts: GenerateOptions,
    model_id: str,
    console: _PlainConsole,
) -> str:
    """Regenerate the most recent user turn."""
    if not session.messages:
        console.show_status("Nothing to retry.")
        return "retry"
    if session.messages[-1].role == "assistant":
        session.messages.pop()
    if not session.messages or session.messages[-1].role != "user":
        console.show_status("Nothing to retry.")
        return "retry"
    last_user = session.messages.pop()
    session.add_user_message(last_user.content, metadata=dict(last_user.metadata))
    session.auto_title()
    return _run_turn(deps, session, gen_opts, model_id, console)


def _drop_pending_user_message(session: ChatSession) -> None:
    """Remove the last pending user message after a failed turn."""
    if session.messages and session.messages[-1].role == "user":
        session.messages.pop()


def _parse_history_limit(arg: str, console: Any) -> int | None:
    """Parse `/history [n]` limit, reporting errors to the console."""
    text = arg.strip()
    if not text:
        return 10
    try:
        limit = int(text)
    except ValueError:
        console.show_error("Usage: /history [n]")
        return None
    if limit <= 0:
        console.show_error("History length must be positive.")
        return None
    return limit


def _export_session(session: ChatSession, arg: str, console: Any) -> None:
    """Export the current chat transcript to Markdown."""
    raw_path = arg.strip() or f"mlxs-chat-{session.session_id}.md"
    target = Path(raw_path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(session.to_markdown(), encoding="utf-8")
    console.show_status(f"Transcript exported to {target}")


def _supports_interactive_shell() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_line(prompt: str | None = None) -> str | None:
    """Read one line from stdin. Returns None on EOF."""
    try:
        if prompt and sys.stdin.isatty():
            return input(prompt)
        return input()
    except EOFError:
        return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
