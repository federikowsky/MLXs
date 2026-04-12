from __future__ import annotations

from pathlib import Path

from mlxs.chat.present.chrome import (
    build_composer_context_fragments,
    build_composer_shortcuts_fragments,
    build_footer_left_fragments,
    build_footer_right_fragments,
    build_header_fragments,
    display_path,
    short_model_name,
    truncate_text,
)


def _join(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _, text in fragments)


def test_build_header_fragments_renders_core_shell_identity() -> None:
    rendered = _join(
        build_header_fragments(
            model_id="z-lab/Qwen3.5-2B-PARO",
            title="New chat",
            cwd_label="~/project",
            branch="main",
            turns=3,
            system_on=False,
        )
    )

    assert "MLXs" in rendered
    assert "Qwen3.5-2B-PARO" in rendered
    assert "~/project" in rendered
    assert "main" in rendered
    assert "3 turns" in rendered


def test_build_composer_context_fragments_for_commands_mentions_and_idle() -> None:
    idle = _join(
        build_composer_context_fragments(
            text="",
            state="idle",
            title="New chat",
            system_on=False,
        )
    )
    command = _join(
        build_composer_context_fragments(
            text="/retry",
            state="idle",
            title="New chat",
            system_on=False,
            command_context="/retry · Regenerate the last user turn.",
        )
    )
    mention = _join(
        build_composer_context_fragments(
            text="@src/server.py",
            state="idle",
            title="New chat",
            system_on=False,
            mention_paths=["src/server.py"],
        )
    )

    assert "working in New chat" in idle
    assert "/retry" in command
    assert "1 file reference(s) ready" in mention


def test_footer_helpers_and_small_utilities_preserve_existing_strings() -> None:
    left = _join(
        build_footer_left_fragments(
            model_id="z-lab/Qwen3.5-2B-PARO",
            branch="main",
            turns=0,
            system_on=False,
            max_tokens=256,
            temperature=0.7,
        )
    )
    right = _join(build_footer_right_fragments(state="generating", detail="Streaming reply"))
    shortcuts = _join(build_composer_shortcuts_fragments(state="idle"))

    assert "0 turns" in left
    assert "max 256" in left
    assert "Streaming reply" in right
    assert "Streaming" in right
    assert "/ commands" in shortcuts
    assert "Enter newline" in shortcuts
    assert "Ctrl+J send" in shortcuts
    assert "Ctrl+↑/↓ switch" in shortcuts
    assert short_model_name("z-lab/Qwen3.5-2B-PARO") == "Qwen3.5-2B-PARO"
    assert truncate_text("abcdef", 5) == "ab..."
    assert display_path(Path.home()) == "~"
