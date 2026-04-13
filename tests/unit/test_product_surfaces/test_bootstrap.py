from __future__ import annotations

from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.bootstrap import _runtime_model_config, _serving_completion_batch_size


def test_serving_completion_batch_size_uses_batch_config() -> None:
    config = AppConfig()
    config = config.model_copy(
        update={
            "batch": config.batch.model_copy(
                update={
                    "completion_batch_size": 4,
                    "max_batch_size": 8,
                }
            )
        }
    )

    assert _serving_completion_batch_size(config) == 4


def test_runtime_model_config_forces_preload_when_lazy_enabled() -> None:
    config = AppConfig()

    runtime_model = _runtime_model_config(config)

    assert runtime_model.preload is True
    assert runtime_model.model_path == config.model.model_path


def test_runtime_model_config_preserves_explicit_eager_model_config() -> None:
    config = AppConfig().model_copy(
        update={
            "model": AppConfig().model.model_copy(update={"lazy_load": False, "preload": False})
        }
    )

    runtime_model = _runtime_model_config(config)

    assert runtime_model is config.model
