from __future__ import annotations

from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.bootstrap import _serving_completion_batch_size


def test_serving_completion_batch_size_is_currently_serialized() -> None:
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

    assert _serving_completion_batch_size(config) == 1
