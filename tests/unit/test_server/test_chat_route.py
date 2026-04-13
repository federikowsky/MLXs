"""Route-level tests for /v1/chat/completions multimodal serving scope."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
from starlette.testclient import TestClient

from mlxs._types import FinishReason, TokenEvent
from mlxs.config.schema import AppConfig
from mlxs.server.app import create_app
import mlxs.server.routes.chat as chat_route


def _make_data_url(mime_type: str, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class _RouteTokenizer:
    eos_token_id = 2
    vocab_size = 4096

    def apply_chat_template(
        self,
        messages: list[dict[str, str | None]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **_kwargs: object,
    ) -> str | list[int]:
        prompt = "\n".join(str(message.get("content") or "") for message in messages)
        if add_generation_prompt:
            prompt = f"{prompt}\nAssistant:"
        return self.encode(prompt) if tokenize else prompt

    def encode(self, text: str) -> list[int]:
        tokens = [101]
        if "<image>" in text:
            tokens.append(201)
        if "<video>" in text:
            tokens.append(301)
        if "<audio>" in text:
            tokens.append(401)
        tokens.append(102)
        return tokens

    def decode(self, token_ids: list[int] | int) -> str:
        del token_ids
        return "ok"


def _make_deps(model_type: str = "qwen3_5") -> tuple[SimpleNamespace, list[dict[str, object]]]:
    config = AppConfig()
    calls: list[dict[str, object]] = []

    def generate_fn(model, tokenizer, prompt, options, *, input_embeddings=None, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        calls.append(
            {
                "model_type": getattr(model, "model_type", ""),
                "prompt": prompt,
                "options": options,
                "input_embeddings": input_embeddings,
            }
        )
        return iter(
            [
                TokenEvent(
                    token_id=1,
                    text="ok",
                    finish_reason=FinishReason.STOP,
                    prompt_tokens=5,
                    generation_tokens=1,
                )
            ]
        )

    deps = SimpleNamespace(
        config=config,
        model=SimpleNamespace(model_type=model_type),
        tokenizer=_RouteTokenizer(),
        prompt_cache=MagicMock(),
        metrics=MagicMock(),
        generate_fn=generate_fn,
    )
    return deps, calls


def test_chat_completions_routes_images_through_real_handler(
    monkeypatch,
) -> None:
    deps, calls = _make_deps()
    media_calls: list[dict[str, object]] = []

    monkeypatch.setattr(chat_route, "load_image", lambda item: f"loaded:{item.mime_type}")

    def fake_process_media_inputs(
        model,
        images,
        input_ids,
        *,
        image_max_pixels=None,
        image_min_pixels=None,
    ):  # type: ignore[no-untyped-def]
        media_calls.append(
            {
                "model_type": model.model_type,
                "images": list(images),
                "shape": tuple(int(dim) for dim in input_ids.shape),
                "image_max_pixels": image_max_pixels,
                "image_min_pixels": image_min_pixels,
            }
        )
        return input_ids, mx.ones((input_ids.shape[1], 2)), "image-hash"

    monkeypatch.setattr(chat_route, "process_media_inputs", fake_process_media_inputs)

    client = TestClient(create_app(deps))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlxs",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _make_data_url("image/jpeg", b"image-bytes"),
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert media_calls == [
        {
            "model_type": "qwen3_5",
            "images": ["loaded:image/jpeg"],
            "shape": (1, 3),
            "image_max_pixels": None,
            "image_min_pixels": None,
        }
    ]
    assert len(calls) == 1
    assert tuple(int(dim) for dim in calls[0]["input_embeddings"].shape) == (3, 2)


def test_chat_completions_routes_supported_video_through_real_handler(
    monkeypatch,
) -> None:
    deps, calls = _make_deps(model_type="qwen3_5")
    video_calls: list[dict[str, object]] = []

    def fake_process_video_inputs(
        model,
        video_items,
        input_ids,
        *,
        image_max_pixels=None,
    ):  # type: ignore[no-untyped-def]
        video_calls.append(
            {
                "model_type": model.model_type,
                "video_count": len(video_items),
                "shape": tuple(int(dim) for dim in input_ids.shape),
                "image_max_pixels": image_max_pixels,
            }
        )
        return input_ids, mx.ones((input_ids.shape[1], 4)), "video-hash"

    monkeypatch.setattr(chat_route, "process_video_inputs", fake_process_video_inputs)

    client = TestClient(create_app(deps))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlxs",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video"},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": _make_data_url("video/mp4", b"video-bytes"),
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert video_calls == [
        {
            "model_type": "qwen3_5",
            "video_count": 1,
            "shape": (1, 3),
            "image_max_pixels": None,
        }
    ]
    assert len(calls) == 1
    assert tuple(int(dim) for dim in calls[0]["input_embeddings"].shape) == (3, 4)


def test_chat_completions_stream_frames_sse_once() -> None:
    deps, _calls = _make_deps()
    client = TestClient(create_app(deps))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mlxs",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "stream please"}],
                }
            ],
        },
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert lines[0].startswith("data: {")
    assert not lines[0].startswith("data: data:")
    assert lines[-1] == "data: [DONE]"


def test_chat_completions_rejects_audio_inputs_in_serving_scope() -> None:
    deps, calls = _make_deps()
    client = TestClient(create_app(deps))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlxs",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"audio-bytes").decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "Audio inputs are not supported" in response.json()["error"]["message"]
    assert calls == []


def test_chat_completions_rejects_video_for_unsupported_model_type() -> None:
    deps, calls = _make_deps(model_type="qwen3")
    client = TestClient(create_app(deps))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlxs",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video"},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": _make_data_url("video/mp4", b"video-bytes"),
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "supported only for qwen3_5, qwen3_5_moe" in message
    assert "got qwen3" in message
    assert calls == []


def test_chat_completions_rejects_mixed_image_and_video_requests() -> None:
    deps, calls = _make_deps(model_type="qwen3_5")
    client = TestClient(create_app(deps))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlxs",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _make_data_url("image/jpeg", b"image-bytes"),
                            },
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": _make_data_url("video/mp4", b"video-bytes"),
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "Mixed image and video requests are not supported" in response.json()["error"]["message"]
    assert calls == []
