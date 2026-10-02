"""Tests for dataset revision pinning (``DatasetRef.revision``).

Covers the spec field, the HF provider pinning the requested revision, the
on-disk verification of a local Hub download, the π0.5 norm-stats cache
being partitioned by revision, and the π0.5 runner refusing to train on data
that is not the pinned revision. No network, openpi or GPU: the Hub API and
the openpi subprocesses are faked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import odyssey.runners.models.pi05_train as pi05
from odyssey.engine.records import MissionRun
from odyssey.providers.huggingface import HFDatasetProvider
from odyssey.runners.base import TaskContext
from odyssey.runners.dataset_revision import (
    DatasetRevisionError,
    check_local_dataset_revision,
    is_commit_sha,
    local_dataset_revision,
)
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    DatasetRef,
    DatasetSource,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
)
from odyssey.telemetry import EventPublisher

SHA_A = "a" * 40
SHA_B = "b" * 40
_CFG = "my_pi05_config"


def _local_download(root: Path, sha: str | None) -> Path:
    """A local dataset dir as ``hf download --local-dir`` leaves it."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text('{"fps": 10}', encoding="utf-8")
    if sha is not None:
        meta = root / ".cache" / "huggingface" / "download" / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json.metadata").write_text(f"{sha}\netag\n1.0\n", encoding="utf-8")
    return root


def _local_ref(root: Path, revision: str | None) -> DatasetRef:
    return DatasetRef(source=DatasetSource.LOCAL, ref=str(root), revision=revision)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

def test_revision_defaults_to_none() -> None:
    assert DatasetRef(source=DatasetSource.LOCAL, ref="/d").revision is None


def test_revision_round_trips() -> None:
    ref = DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name", revision=SHA_A)
    assert DatasetRef.model_validate(ref.model_dump()).revision == SHA_A


# ---------------------------------------------------------------------------
# HF provider pins the requested revision
# ---------------------------------------------------------------------------

class _FakeHfApi:
    def __init__(self, sha: str | None) -> None:
        self.sha = sha
        self.calls: list[tuple[str, str | None]] = []

    def dataset_info(self, repo_id: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((repo_id, kwargs.get("revision")))
        return SimpleNamespace(sha=self.sha)


async def test_provider_passes_requested_revision() -> None:
    api = _FakeHfApi(sha=SHA_A)
    resolved = await HFDatasetProvider(api=api).resolve(
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name", revision="v1.0")
    )
    assert api.calls == [("org/name", "v1.0")]
    assert resolved.revision == SHA_A  # the tag resolved to its commit
    assert resolved.metadata["requested_revision"] == "v1.0"


async def test_provider_without_revision_uses_head() -> None:
    api = _FakeHfApi(sha=SHA_B)
    resolved = await HFDatasetProvider(api=api).resolve(
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name")
    )
    assert api.calls == [("org/name", None)]
    assert resolved.revision == SHA_B


async def test_provider_falls_back_to_requested_revision_without_sha() -> None:
    resolved = await HFDatasetProvider(api=_FakeHfApi(sha=None)).resolve(
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name", revision=SHA_A)
    )
    assert resolved.revision == SHA_A


# ---------------------------------------------------------------------------
# On-disk verification of a local Hub download
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [(SHA_A, True), ("abc1234", True), ("ABC1234", True), ("main", False),
     ("v1.0", False), ("abc12", False)],
)
def test_is_commit_sha(value: str, expected: bool) -> None:
    assert is_commit_sha(value) is expected


def test_local_revision_read_from_download_metadata(tmp_path: Path) -> None:
    assert local_dataset_revision(_local_download(tmp_path / "d", SHA_A)) == SHA_A


def test_local_revision_none_without_metadata(tmp_path: Path) -> None:
    assert local_dataset_revision(_local_download(tmp_path / "d", None)) is None


def test_local_revision_none_for_garbage_metadata(tmp_path: Path) -> None:
    assert local_dataset_revision(_local_download(tmp_path / "d", "not-a-sha")) is None


def test_check_passes_on_matching_revision(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    check_local_dataset_revision("t", _local_ref(root, SHA_A))


def test_check_accepts_abbreviated_sha(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    check_local_dataset_revision("t", _local_ref(root, SHA_A[:7]))


def test_check_fails_on_different_revision(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_B)
    with pytest.raises(DatasetRevisionError, match="downloaded from commit"):
        check_local_dataset_revision("t", _local_ref(root, SHA_A))


@pytest.mark.parametrize("revision", [None, "main"])
def test_check_skips_undeclared_or_branch(tmp_path: Path, revision: str | None) -> None:
    root = _local_download(tmp_path / "d", SHA_B)
    check_local_dataset_revision("t", _local_ref(root, revision))


def test_check_skips_when_not_verifiable(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", None)
    check_local_dataset_revision("t", _local_ref(root, SHA_A))


# ---------------------------------------------------------------------------
# π0.5 norm-stats cache is partitioned by revision
# ---------------------------------------------------------------------------

def _write_stats(cache: Path, repo_id: str) -> None:
    (cache / _CFG / repo_id).mkdir(parents=True)
    (cache / _CFG / repo_id / "norm_stats.json").write_text("{}")


def test_new_revision_of_same_repo_id_misses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale-stats hazard: same repo_id, new data version.

    Stats computed for revision A must not satisfy a run pinned to revision B,
    while a later run on A still reuses its own stats.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    cache_a, hit = pi05._link_norm_stats_cache(run_a, _CFG, "my_dataset", SHA_A)
    assert cache_a is not None and hit is False
    _write_stats(cache_a, "my_dataset")

    run_b = tmp_path / "run_b"
    run_b.mkdir()
    cache_b, hit_b = pi05._link_norm_stats_cache(run_b, _CFG, "my_dataset", SHA_B)
    assert hit_b is False
    assert cache_b != cache_a

    run_a2 = tmp_path / "run_a2"
    run_a2.mkdir()
    _, hit_a2 = pi05._link_norm_stats_cache(run_a2, _CFG, "my_dataset", SHA_A)
    assert hit_a2 is True


def test_unpinned_runs_keep_the_shared_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a revision the cache layout is unchanged (existing caches hit)."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    run = tmp_path / "run"
    run.mkdir()
    cache, _ = pi05._link_norm_stats_cache(run, _CFG, "my_dataset")
    assert cache == tmp_path / "home" / ".odyssey" / "pi05_assets" / _CFG


def test_revision_with_slash_is_path_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = pi05._norm_stats_cache_root(_CFG, "refs/pr/1")
    assert root.parent == tmp_path / "home" / ".odyssey" / "pi05_assets"


# ---------------------------------------------------------------------------
# π0.5 runner refuses data that is not the pinned revision
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        pass


def _context(tmp_path: Path, dataset: DatasetRef) -> TaskContext:
    train = TrainingTask(
        name="finetune",
        training_type=TrainingType.DEMONSTRATION,
        agent_id="pilot",
        dataset=dataset,
        config={"runner": "pi05", "config_name": _CFG},
    )
    mission = Mission(
        metadata=MissionMetadata(name="msn"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(id="pilot", role=AgentRole.PILOT, model=HFModelRef(base="m/m"))
            ],
        ),
        tasks=[
            train,
            EvaluationTask(name="bench", evaluation_type=EvaluationType.CUSTOM,
                           benchmark_name="b"),
        ],
    )
    run = MissionRun.from_spec(mission)
    return TaskContext(
        task=run.tasks[0],
        mission=run,
        publisher=_NullPublisher(),
        output_dir=tmp_path / "out",
        agent=mission.robot.agents[0],
    )


@pytest.fixture
def fake_openpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """openpi scripts on disk + a subprocess stub that writes a checkpoint."""
    repo = tmp_path / "openpi"
    (repo / "scripts").mkdir(parents=True)
    for script in ("train.py", "compute_norm_stats.py"):
        (repo / "scripts" / script).write_text("")
    monkeypatch.setenv("OPENPI_REPO_PATH", str(repo))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    launched: list[str] = []

    async def _fake_subprocess(ctx: TaskContext, spec: Any) -> int:
        launched.append(Path(spec.script_path).name)
        (Path(spec.cwd) / "checkpoints" / _CFG / "exp" / "100").mkdir(parents=True,
                                                                      exist_ok=True)
        return 0

    monkeypatch.setattr(pi05, "run_training_subprocess", _fake_subprocess)
    return launched


def test_runner_records_verified_revision(tmp_path: Path, fake_openpi: list[str]) -> None:
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    result = asyncio.run(pi05.Pi05Runner().run(_context(tmp_path, _local_ref(root, SHA_A))))
    assert result["dataset_revision"] == SHA_A
    assert fake_openpi == ["compute_norm_stats.py", "train.py"]


def test_runner_rejects_wrong_local_revision(tmp_path: Path, fake_openpi: list[str]) -> None:
    root = _local_download(tmp_path / "my_dataset", SHA_B)
    with pytest.raises(DatasetRevisionError):
        asyncio.run(pi05.Pi05Runner().run(_context(tmp_path, _local_ref(root, SHA_A))))
    assert fake_openpi == []  # nothing launched


def test_runner_rejects_pinned_hub_dataset(tmp_path: Path, fake_openpi: list[str]) -> None:
    """openpi can't load a hub dataset at a revision: refuse rather than train on HEAD."""
    ref = DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name", revision=SHA_A)
    with pytest.raises(DatasetRevisionError, match="use source: local"):
        asyncio.run(pi05.Pi05Runner().run(_context(tmp_path, ref)))
    assert fake_openpi == []


def test_runner_unpinned_hub_dataset_still_runs(
    tmp_path: Path, fake_openpi: list[str]
) -> None:
    ref = DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name")
    result = asyncio.run(pi05.Pi05Runner().run(_context(tmp_path, ref)))
    assert result["dataset_revision"] is None
