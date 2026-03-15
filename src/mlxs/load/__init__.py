"""Model loading module — weights, tokenizer, registry (§9, FR1, FR11, AC10)."""

from mlxs.load.registry import MODEL_REGISTRY

__all__ = ["MODEL_REGISTRY", "load_model", "load_model_and_tokenizer", "load_tokenizer"]


def __getattr__(name: str):
    if name in {"load_model", "load_model_and_tokenizer", "load_tokenizer"}:
        from mlxs.load.loader import load_model, load_model_and_tokenizer, load_tokenizer

        mapping = {
            "load_model": load_model,
            "load_model_and_tokenizer": load_model_and_tokenizer,
            "load_tokenizer": load_tokenizer,
        }
        return mapping[name]
    raise AttributeError(name)
