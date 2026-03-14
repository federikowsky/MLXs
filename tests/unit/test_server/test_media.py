"""Tests for server/media.py — media extraction (FR12, §7.4)."""

from __future__ import annotations

import base64

import pytest

from mlxs._errors import InvalidPromptError
from mlxs.server.media import MediaItem, extract_media_from_messages, load_image


def _make_b64_image(fmt: str = "jpeg") -> str:
    """Create a minimal base64 data URL for testing."""
    # Minimal 1x1 pixel JPEG (smallest valid JPEG)
    if fmt == "jpeg":
        # Minimal valid JPEG bytes
        raw = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
            0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C,
            0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
            0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
            0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
            0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
            0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4,
            0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
            0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF,
            0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F,
            0x00, 0x7B, 0x40, 0x1B, 0xFF, 0xD9,
        ])
    else:
        raw = b"fake_image_data"

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/{fmt};base64,{encoded}"


class TestExtractMedia:
    def test_text_only_messages(self) -> None:
        """Text-only messages pass through unchanged."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        text_msgs, media = extract_media_from_messages(msgs)
        assert len(text_msgs) == 2
        assert len(media) == 0
        assert text_msgs[0]["content"] == "You are helpful."
        assert text_msgs[1]["content"] == "Hello"

    def test_image_url_extraction(self) -> None:
        """Image data URL is extracted into MediaItem."""
        data_url = _make_b64_image()
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        text_msgs, media = extract_media_from_messages(msgs)
        assert len(media) == 1
        assert media[0].media_type == "image"
        assert media[0].mime_type == "image/jpeg"
        assert len(media[0].data) > 0
        # Text message should have placeholder
        assert "<image>" in text_msgs[0]["content"]

    def test_multiple_images(self) -> None:
        """Multiple images from same message are extracted."""
        data_url = _make_b64_image()
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Compare these:"},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        text_msgs, media = extract_media_from_messages(msgs)
        assert len(media) == 2

    def test_non_data_url_raises(self) -> None:
        """Non-base64 image URLs raise InvalidPromptError."""
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ]},
        ]
        with pytest.raises(InvalidPromptError, match="base64"):
            extract_media_from_messages(msgs)

    def test_unsupported_content_type_raises(self) -> None:
        """Unsupported content types raise InvalidPromptError."""
        msgs = [
            {"role": "user", "content": [
                {"type": "file", "file": {"url": "data:..."}},
            ]},
        ]
        with pytest.raises(InvalidPromptError, match="Unsupported"):
            extract_media_from_messages(msgs)

    def test_audio_not_yet_supported(self) -> None:
        """Audio content type raises not-yet-supported error."""
        msgs = [
            {"role": "user", "content": [
                {"type": "audio", "audio": {}},
            ]},
        ]
        with pytest.raises(InvalidPromptError, match="not yet supported"):
            extract_media_from_messages(msgs)

    def test_none_content_passthrough(self) -> None:
        """Messages with None content pass through."""
        msgs = [{"role": "assistant", "content": None}]
        text_msgs, media = extract_media_from_messages(msgs)
        assert len(text_msgs) == 1
        assert len(media) == 0

    def test_invalid_data_url(self) -> None:
        """Malformed data URL raises InvalidPromptError."""
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:badformat"}},
            ]},
        ]
        with pytest.raises(InvalidPromptError, match="Invalid data URL"):
            extract_media_from_messages(msgs)


class TestLoadImage:
    def test_load_valid_jpeg(self) -> None:
        """Valid JPEG data loads as PIL Image."""
        pytest.importorskip("PIL")
        data_url = _make_b64_image("jpeg")
        _, media = extract_media_from_messages([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ])
        img = load_image(media[0])
        assert img is not None
        assert img.mode == "RGB"

    def test_load_corrupt_image(self) -> None:
        """Corrupt image data raises InvalidPromptError."""
        pytest.importorskip("PIL")
        item = MediaItem(media_type="image", data=b"not_an_image", mime_type="image/jpeg")
        with pytest.raises(InvalidPromptError, match="Failed to load"):
            load_image(item)
