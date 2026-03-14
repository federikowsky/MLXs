"""Chat package — template, session, store and controller for multi-turn chat.

Public API:
    - template: prompt building, stop-token discovery, output sanitization
    - session: data models for chat sessions/messages
    - store: persistent session storage
    - controller: UI-agnostic conversation orchestrator
"""

from __future__ import annotations
