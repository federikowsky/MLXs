"""Tests for Layer 3 engine-local admission policy."""

from __future__ import annotations

from mlxs.advanced_engines.admission import AdmissionDecision, EngineAdmissionPolicy


def test_engine_admission_policy_admit() -> None:
    policy = EngineAdmissionPolicy(max_active=4, max_pending=8)
    assert policy.decide(active_count=1, pending_count=2) is AdmissionDecision.ADMIT


def test_engine_admission_policy_defers_when_active_at_capacity() -> None:
    policy = EngineAdmissionPolicy(max_active=2, max_pending=8)
    assert policy.decide(active_count=2, pending_count=1) is AdmissionDecision.DEFER


def test_engine_admission_policy_rejects_when_pending_at_capacity() -> None:
    policy = EngineAdmissionPolicy(max_active=2, max_pending=3)
    assert policy.decide(active_count=1, pending_count=3) is AdmissionDecision.REJECT
