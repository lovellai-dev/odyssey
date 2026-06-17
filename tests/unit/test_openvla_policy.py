"""Tests for OpenVLA inference policy helpers in odyssey.runners.openvla.

Pure Python — no GPU or transformers needed. The functions under test
(_resolve_base_model, _find_image_key, make_openvla_policy) live in
openvla.py alongside the training runner. We import them through the
package so existing modules are already initialized.

NOTE: running this file in isolation (``pytest tests/unit/test_openvla_policy.py``)
hits the pre-existing engine ↔ runners circular import. Run via ``pytest tests/``
or alongside any other test file that loads the runners package first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odyssey.runners.openvla import (
    _find_image_key,
    _resolve_base_model,
    make_openvla_policy,
)

# ---------------------------------------------------------------------------
# _resolve_base_model
# ---------------------------------------------------------------------------


def test_resolve_base_model_reads_adapter_config(tmp_path: Path) -> None:
    config = {"base_model_name_or_path": "openvla/openvla-7b"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    assert _resolve_base_model(tmp_path) == "openvla/openvla-7b"


def test_resolve_base_model_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"adapter_config\.json"):
        _resolve_base_model(tmp_path)


def test_resolve_base_model_missing_key_raises(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        _resolve_base_model(tmp_path)


# ---------------------------------------------------------------------------
# _find_image_key
# ---------------------------------------------------------------------------


def test_find_image_key_preferred() -> None:
    obs = {"agentview_image": [1, 2, 3], "other": [4, 5, 6]}
    assert _find_image_key(obs, "agentview_image") == "agentview_image"


def test_find_image_key_fallback() -> None:
    obs = {"frontview_image": [1, 2, 3], "joint_pos": [0.1]}
    assert _find_image_key(obs, "agentview_image") == "frontview_image"


def test_find_image_key_no_image_raises() -> None:
    obs = {"joint_pos": [0.1], "gripper_state": [0.5]}
    with pytest.raises(KeyError, match="No image key found"):
        _find_image_key(obs, "agentview_image")


# ---------------------------------------------------------------------------
# make_openvla_policy — dependency guard
# ---------------------------------------------------------------------------


def test_make_openvla_policy_raises_without_deps(tmp_path: Path) -> None:
    """Without transformers/peft/torch, make_openvla_policy should raise
    NotImplementedError with 'policy' in the message."""
    # Write a valid adapter_config so we get past path checks
    config = {"base_model_name_or_path": "openvla/openvla-7b"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="policy"):
        make_openvla_policy(tmp_path)
