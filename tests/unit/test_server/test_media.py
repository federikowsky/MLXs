"""Tests for server/media.py — media extraction (FR12, §7.4)."""

from __future__ import annotations

import base64

import mlx.core as mx
import pytest

from mlxs._errors import InvalidPromptError
from mlxs.server.media import (
    MediaItem,
    _preprocess_images,
    extract_media_from_messages,
    load_image,
    process_video_inputs,
)


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
        _text_msgs, media = extract_media_from_messages(msgs)
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

    def test_audio_extraction(self) -> None:
        """input_audio content type is extracted into MediaItem."""
        audio_b64 = base64.b64encode(b"fake_audio_data").decode("ascii")
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this sound?"},
                {"type": "input_audio", "input_audio": {
                    "data": audio_b64, "format": "wav",
                }},
            ]},
        ]
        text_msgs, media = extract_media_from_messages(msgs)
        assert len(media) == 1
        assert media[0].media_type == "audio"
        assert media[0].mime_type == "audio/wav"
        assert media[0].data == b"fake_audio_data"
        assert "<audio>" in text_msgs[0]["content"]

    def test_video_url_extraction(self) -> None:
        """video_url data URL is extracted into MediaItem."""
        video_b64 = base64.b64encode(b"fake_video_data").decode("ascii")
        data_url = f"data:video/mp4;base64,{video_b64}"
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "What happens in this video?"},
                {"type": "video_url", "video_url": {"url": data_url}},
            ]},
        ]
        text_msgs, media = extract_media_from_messages(msgs)
        assert len(media) == 1
        assert media[0].media_type == "video"
        assert media[0].mime_type == "video/mp4"
        assert "<video>" in text_msgs[0]["content"]

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


class _VideoModel:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        self.calls: list[dict[str, object]] = []

    def prepare_inputs(self, input_ids, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return input_ids, None


def test_preprocess_images_routes_qwen3_5_to_qwen_vl(monkeypatch: pytest.MonkeyPatch) -> None:
    """qwen3_5 image routing uses the Qwen VL preprocessing branch."""
    calls: list[tuple[object, object, object, object]] = []

    def fake_preprocess_qwen_vl(images, vision_cfg, max_pixels=None, min_pixels=None):  # type: ignore[no-untyped-def]
        calls.append((images, vision_cfg, max_pixels, min_pixels))
        return "pixel_values", "grid"

    monkeypatch.setattr(
        "mlxs.models.vision.image_processing.preprocess_qwen_vl",
        fake_preprocess_qwen_vl,
    )

    pixel_values, extra_kwargs = _preprocess_images("qwen3_5", [object()])

    assert pixel_values == "pixel_values"
    assert extra_kwargs == {"image_grid_thw": "grid"}
    assert calls and calls[0][1] == {
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }


@pytest.mark.parametrize("model_type", ["pixtral", "mistral3"])
def test_preprocess_images_routes_pixtral_family_to_pixtral_branch(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    """Pixtral-style families keep using preprocess_pixtral in server/media."""
    calls: list[tuple[object, object]] = []

    def fake_preprocess_pixtral(images, max_pixels=None):  # type: ignore[no-untyped-def]
        calls.append((images, max_pixels))
        return "pixel_values", [(4, 4)]

    monkeypatch.setattr(
        "mlxs.models.vision.image_processing.preprocess_pixtral",
        fake_preprocess_pixtral,
    )

    pixel_values, extra_kwargs = _preprocess_images(model_type, [object()])

    assert pixel_values == "pixel_values"
    assert extra_kwargs == {"image_sizes": [(4, 4)]}
    assert len(calls) == 1
    assert calls[0][1] is None


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "qwen3_moe"])
def test_preprocess_images_routes_canonical_qwen_families_to_qwen_vl(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    """Canonical unified Qwen family keys use the Qwen VL preprocessing branch."""
    calls: list[tuple[object, object, object, object]] = []

    def fake_preprocess_qwen_vl(images, vision_cfg, max_pixels=None, min_pixels=None):  # type: ignore[no-untyped-def]
        calls.append((images, vision_cfg, max_pixels, min_pixels))
        return "pixel_values", "grid"

    monkeypatch.setattr(
        "mlxs.models.vision.image_processing.preprocess_qwen_vl",
        fake_preprocess_qwen_vl,
    )

    pixel_values, extra_kwargs = _preprocess_images(model_type, [object()])

    assert pixel_values == "pixel_values"
    assert extra_kwargs == {"image_grid_thw": "grid"}
    assert calls and calls[0][1] == {
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }


def test_preprocess_images_routes_kimi_vl_to_kimi_branch() -> None:
    """kimi_vl uses dedicated MoonViT preprocessing and returns a patch grid."""
    pytest.importorskip("PIL")
    from PIL import Image

    pixel_values, extra_kwargs = _preprocess_images(
        "kimi_vl",
        [Image.new("RGB", (14, 14), color=(128, 128, 128))],
    )

    assert tuple(int(dim) for dim in pixel_values.shape) == (1, 14, 14, 3)
    assert "image_grid_thw" in extra_kwargs
    assert mx.array_equal(extra_kwargs["image_grid_thw"], mx.array([[1, 1]], dtype=mx.int32))


def test_preprocess_images_routes_lfm2_to_standard_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lfm2 keeps using preprocess_standard; patchification stays model-local."""
    calls: list[list[object]] = []

    def fake_preprocess_standard(images):  # type: ignore[no-untyped-def]
        calls.append(images)
        return "pixel_values"

    monkeypatch.setattr(
        "mlxs.models.vision.image_processing.preprocess_standard",
        fake_preprocess_standard,
    )

    pixel_values, extra_kwargs = _preprocess_images("lfm2", [object()])

    assert pixel_values == "pixel_values"
    assert extra_kwargs == {}
    assert len(calls) == 1


def test_preprocess_images_routes_qwen3_5_moe_to_qwen_vl_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """qwen3_5_moe uses the same canonical Qwen VL image preprocessing branch."""
    qwen_calls: list[tuple[object, object, object, object]] = []

    def fake_preprocess_qwen_vl(images, vision_cfg, max_pixels=None, min_pixels=None):  # type: ignore[no-untyped-def]
        qwen_calls.append((images, vision_cfg, max_pixels, min_pixels))
        return "pixel_values", "grid"

    monkeypatch.setattr(
        "mlxs.models.vision.image_processing.preprocess_qwen_vl",
        fake_preprocess_qwen_vl,
    )

    pixel_values, extra_kwargs = _preprocess_images("qwen3_5_moe", [object()])

    assert pixel_values == "pixel_values"
    assert extra_kwargs == {"image_grid_thw": "grid"}
    assert qwen_calls and qwen_calls[0][1] == {
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }


def test_process_video_inputs_uses_video_kwargs_for_qwen3_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """qwen3_5 video routing uses video_pixel_values + video_grid_thw."""
    monkeypatch.setattr(
        "mlxs.models.vision.video_processing.extract_video_frames",
        lambda *args, **kwargs: ["frame"],
    )
    monkeypatch.setattr(
        "mlxs.models.vision.video_processing.preprocess_video_qwen_vl",
        lambda *args, **kwargs: ("video_pixels", "video_grid"),
    )

    model = _VideoModel("qwen3_5")
    input_ids_out, input_embeddings, media_hash = process_video_inputs(
        model,
        [MediaItem(media_type="video", data=b"video-bytes", mime_type="video/mp4")],
        input_ids=mx.array([[1, 2]]),
    )

    assert mx.array_equal(input_ids_out, mx.array([[1, 2]]))
    assert input_embeddings is None
    assert media_hash is not None
    assert model.calls == [
        {
            "video_pixel_values": "video_pixels",
            "video_grid_thw": "video_grid",
        }
    ]


def test_process_video_inputs_uses_video_kwargs_for_qwen3_5_moe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """qwen3_5_moe shares the dedicated Qwen3.5 video placeholder path."""
    monkeypatch.setattr(
        "mlxs.models.vision.video_processing.extract_video_frames",
        lambda *args, **kwargs: ["frame"],
    )
    monkeypatch.setattr(
        "mlxs.models.vision.video_processing.preprocess_video_qwen_vl",
        lambda *args, **kwargs: ("video_pixels", "video_grid"),
    )

    model = _VideoModel("qwen3_5_moe")
    input_ids_out, input_embeddings, media_hash = process_video_inputs(
        model,
        [MediaItem(media_type="video", data=b"video-bytes", mime_type="video/mp4")],
        input_ids=mx.array([[1, 2]]),
    )

    assert mx.array_equal(input_ids_out, mx.array([[1, 2]]))
    assert input_embeddings is None
    assert media_hash is not None
    assert model.calls == [
        {
            "video_pixel_values": "video_pixels",
            "video_grid_thw": "video_grid",
        }
    ]


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "qwen3_moe"])
def test_process_video_inputs_uses_image_kwargs_for_canonical_qwen_families(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    """Canonical unified Qwen family keys use the shared Qwen VL video branch."""
    monkeypatch.setattr(
        "mlxs.models.vision.video_processing.extract_video_frames",
        lambda *args, **kwargs: ["frame"],
    )
    monkeypatch.setattr(
        "mlxs.models.vision.video_processing.preprocess_video_qwen_vl",
        lambda *args, **kwargs: ("video_pixels", "video_grid"),
    )

    model = _VideoModel(model_type)
    input_ids_out, input_embeddings, media_hash = process_video_inputs(
        model,
        [MediaItem(media_type="video", data=b"video-bytes", mime_type="video/mp4")],
        input_ids=mx.array([[1, 2]]),
    )

    assert mx.array_equal(input_ids_out, mx.array([[1, 2]]))
    assert input_embeddings is None
    assert media_hash is not None
    assert model.calls == [
        {
            "pixel_values": "video_pixels",
            "image_grid_thw": "video_grid",
        }
    ]
