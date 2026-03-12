"""Model architectures — per-architecture model classes (§7, §9, AC17).

Models are registered via the ``MODEL_REGISTRY`` dict and discovered by
model_type from config.json. Adding a new architecture requires only a
new file in this package and a registry entry — no changes to generate,
cache, batch, or server.
"""
