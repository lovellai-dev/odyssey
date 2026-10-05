"""Tests for dataset revision pinning (``DatasetRef.revision``).

Covers the spec field, the HF provider pinning the requested revision, the
file-by-file verification of a local Hub download, the π0.5 norm-stats
cache being partitioned by a verified revision, and the π0.5 runner refusing
to train on data that is not the pinned revision — checked in the directory
the #102 pre-flight probe says training loads. No network, openpi or GPU: the
Hub API, the openpi config probe and the openpi subprocesses are faked.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import odyssey.runners.models.gr00t_train as gr00t_train
import odyssey.runners.models.openvla_train as openvla_train
import odyssey.runners.models.pi05_train as pi05
from odyssey.engine.records import MissionRun
from odyssey.providers.huggingface import HFDatasetProvider
from odyssey.runners.base import TaskContext
from odyssey.runners.dataset_revision import (
    DatasetRevisionError,
    check_local_dataset_revision,
    dataset_files,
    is_commit_sha,
    verify_local_dataset_revision,
)
from odyssey.runners.models.gr00t_train import GR00TRunner
from odyssey.runners.models.openpi_dataset import (
    DatasetMismatchError,
    OpenpiTrainingConfig,
)
from odyssey.runners.models.openvla_train import OpenVLARunner
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


_EPISODE = "data/chunk-000/episode_000000.parquet"
_VIDEO = "videos/chunk-000/observation.images.wrist/episode_000000.mp4"


def _info(version: str = "v2.1", **overrides: Any) -> str:
    """A minimal LeRobot v2.x ``meta/info.json``: one episode, one camera."""
    return json.dumps({
        "codebase_version": version,
        "fps": 10,
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.wrist": {"dtype": "video"},
            "observation.state": {"dtype": "float32"},
        },
        **overrides,
    })


# Everything the LeRobot loader reads, per format. v2.0 keeps one aggregate
# stats.json; v2.1 keeps per-episode stats.
_STATS_FILE = {"v2.0": "meta/stats.json", "v2.1": "meta/episodes_stats.jsonl"}
_STATS_CONTENT = {
    "v2.0": '{"observation.state": {"mean": [0.0]}}',
    "v2.1": '{"episode_index": 0, "stats": {}}\n',
}


def _content(version: str = "v2.1") -> dict[str, str]:
    return {
        "meta/info.json": _info(version),
        "meta/episodes.jsonl": '{"episode_index": 0, "length": 5}\n',
        "meta/tasks.jsonl": '{"task_index": 0, "task": "pick"}\n',
        _STATS_FILE[version]: _STATS_CONTENT[version],
        _EPISODE: "x",
        _VIDEO: "x",
        ".gitattributes": "x",
    }


_CONTENT = _content()
_FILES = tuple(_CONTENT)
_N = len(_FILES)


def _hub_metadata(root: Path, rel: str, sha: str) -> None:
    """The per-file record ``hf download --local-dir`` writes after a file."""
    meta = root / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
    meta.parent.mkdir(parents=True, exist_ok=True)
    mtime = (root / rel).stat().st_mtime
    meta.write_text(f"{sha}\netag\n{mtime}\n", encoding="utf-8")


def _hub_metadata_unchecked(root: Path, rel: str, sha: str) -> None:
    """Metadata for a path that can't be stat'ed (e.g. a dangling symlink)."""
    meta = root / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(f"{sha}\netag\n0\n", encoding="utf-8")


def _write(root: Path, rel: str, content: str = "x") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(content, encoding="utf-8")


def _local_download(root: Path, sha: str | None, version: str = "v2.1") -> Path:
    """A complete local dataset dir as ``hf download --local-dir`` leaves it.

    Every file gets download metadata for ``sha``; ``None`` writes the files
    with no metadata at all (a dataset recorded or copied by hand).
    """
    for rel, content in _content(version).items():
        _write(root, rel, content)
        if sha is not None:
            _hub_metadata(root, rel, sha)
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


async def test_provider_never_records_a_branch_as_content_hash() -> None:
    """Regression: no Hub sha + a branch request used to give content_hash 'hf-sha:main'."""
    resolved = await HFDatasetProvider(api=_FakeHfApi(sha=None)).resolve(
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name", revision="main")
    )
    assert resolved.revision == "main"
    assert resolved.content_hash is None


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


def test_dataset_files_skip_the_hub_cache(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    assert [str(f) for f in dataset_files(root)] == sorted(_FILES)


def test_verify_passes_when_every_file_matches(tmp_path: Path) -> None:
    verify_local_dataset_revision(_local_download(tmp_path / "d", SHA_A), SHA_A)


def test_verify_accepts_abbreviated_sha(tmp_path: Path) -> None:
    verify_local_dataset_revision(_local_download(tmp_path / "d", SHA_A), SHA_A[:7])


def test_verify_fails_on_different_revision(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_B)
    with pytest.raises(
        DatasetRevisionError, match=rf"{_N} problems, {_N} files on disk.*downloaded from commit"
    ):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_on_mixed_revisions(tmp_path: Path) -> None:
    """Regression: the info.json marker says A, but one episode came from B."""
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, _EPISODE, "corrected labels")
    _hub_metadata(root, _EPISODE, SHA_B)
    with pytest.raises(DatasetRevisionError, match=rf"1 problems, {_N} files on disk.*"
                       rf"episode_000000\.parquet: downloaded from commit {SHA_B}"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_on_local_edit(tmp_path: Path) -> None:
    """Regression: a file edited after download keeps A's metadata but not A's data."""
    root = _local_download(tmp_path / "d", SHA_A)
    episode = root / _EPISODE
    episode.write_text("edited", encoding="utf-8")
    later = episode.stat().st_mtime + 60
    os.utime(episode, (later, later))
    with pytest.raises(DatasetRevisionError, match="modified after download"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_tolerates_mtime_jitter(tmp_path: Path) -> None:
    """The Hub's rule allows 1 s of mtime imprecision."""
    root = _local_download(tmp_path / "d", SHA_A)
    episode = root / _EPISODE
    jitter = episode.stat().st_mtime + 0.5
    os.utime(episode, (jitter, jitter))
    verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_on_file_without_metadata(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, "data/chunk-000/episode_000001.parquet")
    with pytest.raises(DatasetRevisionError, match=r"episode_000001\.parquet: no Hub download"):
        verify_local_dataset_revision(root, SHA_A)


@pytest.mark.parametrize("sha", [None, "not-a-sha"])
def test_verify_fails_without_usable_metadata(tmp_path: Path, sha: str | None) -> None:
    root = _local_download(tmp_path / "d", None)
    if sha is not None:
        for rel in _FILES:
            (root / ".cache/huggingface/download" / f"{rel}.metadata").parent.mkdir(
                parents=True, exist_ok=True
            )
            (root / ".cache/huggingface/download" / f"{rel}.metadata").write_text(sha)
    with pytest.raises(DatasetRevisionError, match=f"{_N} problems, {_N} files on disk"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_on_empty_or_missing_dir(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    for root in (tmp_path / "empty", tmp_path / "missing"):
        with pytest.raises(DatasetRevisionError, match="no files to verify"):
            verify_local_dataset_revision(root, SHA_A)


def test_verify_error_caps_the_listed_files(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    for i in range(8):
        _write(root, f"data/extra_{i}.parquet")
    with pytest.raises(
        DatasetRevisionError, match=rf"8 problems, {_N + 8} files on disk.*and 3 more"
    ):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_rejects_empty_commit_line(tmp_path: Path) -> None:
    """Regression: "" is a prefix of every sha and must not count as a match."""
    root = _local_download(tmp_path / "d", SHA_A)
    _hub_metadata(root, _EPISODE, "")
    with pytest.raises(DatasetRevisionError, match=r"\(none recorded\)"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_follows_symlinked_directories(tmp_path: Path) -> None:
    """Regression: data on another volume (data -> /elsewhere) is checked too."""
    root = _local_download(tmp_path / "d", SHA_A)
    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere, "episode_000009.parquet")
    (root / "linked").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(DatasetRevisionError, match=r"linked/episode_000009\.parquet"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_survives_a_symlink_cycle(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    (root / "data" / "loop").symlink_to(root / "data", target_is_directory=True)
    verify_local_dataset_revision(root, SHA_A)


def test_verify_reports_broken_symlink(tmp_path: Path) -> None:
    """Regression: a dangling file symlink is a listed problem, not a bare OSError."""
    root = _local_download(tmp_path / "d", SHA_A)
    (root / "data" / "gone.parquet").symlink_to(tmp_path / "missing.parquet")
    _hub_metadata_unchecked(root, "data/gone.parquet", SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"gone\.parquet: unreadable"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_on_partial_download(tmp_path: Path) -> None:
    """Regression: every file present is A's, but a file training reads is absent."""
    root = _local_download(tmp_path / "d", SHA_A)
    _remove(root, _VIDEO)
    with pytest.raises(DatasetRevisionError, match=r"episode_000000\.mp4: missing"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_on_episode_listed_but_not_downloaded(tmp_path: Path) -> None:
    """episodes.jsonl is the loader's list: an episode it names must be on disk."""
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, "meta/episodes.jsonl",
           '{"episode_index": 0}\n{"episode_index": 1001}\n')
    _hub_metadata(root, "meta/episodes.jsonl", SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"chunk-001/episode_001001\.parquet: missing"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_fails_without_a_lerobot_manifest(tmp_path: Path) -> None:
    """No v2.x manifest: which files training reads is unknown, so is the pin."""
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, "meta/info.json", '{"fps": 10}')
    _hub_metadata(root, "meta/info.json", SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"not a LeRobot v2\.x manifest"):
        verify_local_dataset_revision(root, SHA_A)


def _remove(root: Path, rel: str) -> None:
    """Drop a file and its download metadata, as a partial download leaves it."""
    (root / rel).unlink()
    (root / ".cache/huggingface/download" / f"{rel}.metadata").unlink()


@pytest.mark.parametrize("version", ["v2.0", "v2.1"])
def test_verify_passes_on_complete_snapshot_of_each_format(
    tmp_path: Path, version: str
) -> None:
    verify_local_dataset_revision(_local_download(tmp_path / "d", SHA_A, version), SHA_A)


@pytest.mark.parametrize(
    "version, rel",
    [
        (version, rel)
        for version in ("v2.0", "v2.1")
        for rel in ("meta/tasks.jsonl", "meta/episodes.jsonl", _STATS_FILE[version])
    ],
)
def test_verify_fails_when_loader_metadata_is_missing(
    tmp_path: Path, version: str, rel: str
) -> None:
    """Regression: a missing metadata file makes LeRobot pull meta/ from another revision."""
    root = _local_download(tmp_path / "d", SHA_A, version)
    _remove(root, rel)
    with pytest.raises(DatasetRevisionError, match=rf"{re.escape(rel)}: missing"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_v20_does_not_require_v21_stats(tmp_path: Path) -> None:
    """Each format needs its own stats file only, not the other one's."""
    root = _local_download(tmp_path / "d", SHA_A, "v2.0")
    assert not (root / _STATS_FILE["v2.1"]).exists()
    verify_local_dataset_revision(root, SHA_A)


def test_verify_requires_every_episode_in_total_episodes(tmp_path: Path) -> None:
    """Regression: the loader checks range(total_episodes), even if episodes.jsonl lists fewer."""
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, "meta/info.json", _info(total_episodes=2))
    _hub_metadata(root, "meta/info.json", SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"episode_000001\.parquet: missing"):
        verify_local_dataset_revision(root, SHA_A)


@pytest.mark.parametrize(
    "info",
    [
        _info("v3.0"),
        _info("v1.6"),
        _info(codebase_version=None),
        _info(chunks_size=0),
        _info(data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"),
    ],
    ids=["v3.0", "v1.6", "no-version", "zero-chunks", "v3-template"],
)
def test_verify_rejects_non_v2_manifest(tmp_path: Path, info: str) -> None:
    """Not a v2.x layout: the files training reads are unknown — an error, not a crash."""
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, "meta/info.json", info)
    _hub_metadata(root, "meta/info.json", SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"LeRobot v2\.x"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_rejects_non_dict_features(tmp_path: Path) -> None:
    """Regression: a malformed features block is a revision error, not an AttributeError."""
    root = _local_download(tmp_path / "d", SHA_A)
    _write(root, "meta/info.json", _info(features="x"))
    _hub_metadata(root, "meta/info.json", SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"LeRobot v2\.x"):
        verify_local_dataset_revision(root, SHA_A)


@pytest.mark.parametrize("declared", [SHA_A, SHA_A[:7], SHA_A[:7].upper()])
def test_verify_returns_the_full_lowercase_commit(tmp_path: Path, declared: str) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    assert verify_local_dataset_revision(root, declared) == SHA_A


def test_verify_rejects_two_commits_sharing_the_abbreviated_prefix(tmp_path: Path) -> None:
    """An abbreviated pin must name ONE commit: files from two that share it fail."""
    sha_a2 = SHA_A[:7] + "0" * 33
    root = _local_download(tmp_path / "d", SHA_A)
    _hub_metadata(root, _EPISODE, sha_a2)
    with pytest.raises(DatasetRevisionError, match="files from 2 commits"):
        verify_local_dataset_revision(root, SHA_A[:7])
    # The full sha is contradicted only by the other commit's file.
    with pytest.raises(DatasetRevisionError, match=r"episode_000000\.parquet"):
        verify_local_dataset_revision(root, SHA_A)


def test_verify_skips_in_tree_symlink_alias(tmp_path: Path) -> None:
    """Regression: an alias (videos_latest -> videos) must not fail a valid copy."""
    root = _local_download(tmp_path / "d", SHA_A)
    (root / "videos_latest").symlink_to(root / "videos", target_is_directory=True)
    verify_local_dataset_revision(root, SHA_A)
    assert not any(str(f).startswith("videos_latest") for f in dataset_files(root))


def test_check_reports_verified_sha(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_A)
    assert check_local_dataset_revision("t", _local_ref(root, SHA_A), str(root)) == SHA_A


def test_check_verifies_the_loaded_dir(tmp_path: Path) -> None:
    """The directory training loads is what gets verified, not dataset.ref."""
    declared = _local_download(tmp_path / "declared", SHA_A)
    loaded = _local_download(tmp_path / "loaded", SHA_B)
    with pytest.raises(DatasetRevisionError, match="loaded is not verifiably"):
        check_local_dataset_revision("t", _local_ref(declared, SHA_A), str(loaded))


def test_check_names_task_and_remedy(tmp_path: Path) -> None:
    root = _local_download(tmp_path / "d", SHA_B)
    with pytest.raises(DatasetRevisionError, match=r"task 't'.*rsync -a"):
        check_local_dataset_revision("t", _local_ref(root, SHA_A), str(root))


def test_check_fails_when_sha_pin_is_unverifiable(tmp_path: Path) -> None:
    """A sha pin claims an identity: no metadata means no claim, so refuse."""
    root = _local_download(tmp_path / "d", None)
    with pytest.raises(DatasetRevisionError, match="no Hub download metadata"):
        check_local_dataset_revision("t", _local_ref(root, SHA_A), str(root))


def test_check_fails_when_loaded_dir_is_unknown(tmp_path: Path) -> None:
    """No resolvable training directory (no LeRobot repo_id): can't verify, refuse."""
    root = _local_download(tmp_path / "d", SHA_A)
    with pytest.raises(DatasetRevisionError, match="could not be resolved"):
        check_local_dataset_revision("t", _local_ref(root, SHA_A), None)


def test_check_fails_for_relative_local_ref_with_sha_pin() -> None:
    """Regression: a relative ref used to skip verification silently."""
    ref = DatasetRef(source=DatasetSource.LOCAL, ref="datasets/foo", revision=SHA_A)
    with pytest.raises(DatasetRevisionError, match=r"absolute dataset\.ref"):
        check_local_dataset_revision("t", ref, "/somewhere/foo")


@pytest.mark.parametrize("revision", [None, "main", "v1.0"])
def test_check_undeclared_or_branch_is_unverified(tmp_path: Path, revision: str | None) -> None:
    root = _local_download(tmp_path / "d", SHA_B)
    assert check_local_dataset_revision("t", _local_ref(root, revision), str(root)) is None


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


def _context(
    tmp_path: Path, dataset: DatasetRef, **config: Any
) -> TaskContext:
    train = TrainingTask(
        name="finetune",
        training_type=TrainingType.DEMONSTRATION,
        agent_id="pilot",
        dataset=dataset,
        config={"runner": "pi05", "config_name": _CFG, **config},
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
def fake_openpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """openpi scripts on disk, a fake config probe and a subprocess stub.

    The probe resolves the loaded directory the way openpi does,
    ``HF_LEROBOT_HOME / data.repo_id``, with the registered config's default
    repo_id taken to be ``my_dataset``.
    """
    repo = tmp_path / "openpi"
    (repo / "scripts").mkdir(parents=True)
    for script in ("train.py", "compute_norm_stats.py"):
        (repo / "scripts" / script).write_text("")
    monkeypatch.setenv("OPENPI_REPO_PATH", str(repo))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    launched: list[Any] = []

    async def _probe(
        config_name: str, exp_name: str, overrides: list[str], env: dict[str, str],
        cancel_event: Any = None,
    ) -> OpenpiTrainingConfig:
        repo_id = "my_dataset"
        if "--data.repo-id" in overrides:
            repo_id = overrides[overrides.index("--data.repo-id") + 1]
        home = env.get("HF_LEROBOT_HOME", str(tmp_path / "hf_home"))
        return OpenpiTrainingConfig(16, repo_id, str(Path(home) / repo_id))

    async def _fake_subprocess(ctx: TaskContext, spec: Any) -> int:
        launched.append(spec)
        if Path(spec.script_path).name == "compute_norm_stats.py":
            _write(Path(spec.cwd), f"assets/{_CFG}/my_dataset/norm_stats.json", "{}")
        (Path(spec.cwd) / "checkpoints" / _CFG / "exp" / "100").mkdir(parents=True,
                                                                      exist_ok=True)
        return 0

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _probe)
    monkeypatch.setattr(pi05, "run_training_subprocess", _fake_subprocess)
    return launched


def _cached_stats(tmp_path: Path, cache_dir: str) -> Path:
    """Where a published norm-stats entry lives in the shared cache."""
    return (tmp_path / "home" / ".odyssey" / "pi05_assets" / cache_dir / _CFG
            / "my_dataset" / "norm_stats.json")


def _scripts(launched: list[Any]) -> list[str]:
    return [Path(spec.script_path).name for spec in launched]


def _run(ctx: TaskContext) -> dict[str, Any]:
    result: dict[str, Any] = asyncio.run(pi05.Pi05Runner().run(ctx))
    return result


def test_runner_records_verified_revision(tmp_path: Path, fake_openpi: list[Any]) -> None:
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    result = _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert result["dataset_revision"] == SHA_A
    assert result["dataset_revision_verified"] is True
    assert _scripts(fake_openpi) == ["compute_norm_stats.py", "train.py"]
    assert _cached_stats(tmp_path, f"{_CFG}@{SHA_A}").is_file()  # published after re-check


@pytest.mark.parametrize("declared", [SHA_A[:7], SHA_A[:7].upper()])
def test_runner_records_and_caches_by_the_full_sha(
    tmp_path: Path, fake_openpi: list[Any], declared: str
) -> None:
    """An abbreviated or upper-case pin shares the full sha's cache and record."""
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    result = _run(_context(tmp_path, _local_ref(root, declared)))
    assert result["dataset_revision"] == SHA_A
    assert _cached_stats(tmp_path, f"{_CFG}@{SHA_A}").is_file()


def test_runner_reverifies_before_train(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: data replaced while norm stats run must not train as 'verified'."""
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    launched: list[str] = []

    async def _swaps_an_episode(ctx: TaskContext, spec: Any) -> int:
        launched.append(Path(spec.script_path).name)
        _hub_metadata(root, _EPISODE, SHA_B)  # someone re-downloads mid-run
        return 0

    monkeypatch.setattr(pi05, "run_training_subprocess", _swaps_an_episode)
    with pytest.raises(DatasetRevisionError, match=r"episode_000000\.parquet"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert launched == ["compute_norm_stats.py"]  # train.py never started


def test_rejected_run_publishes_no_stats_for_the_next_run(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: stats computed on swapped data must not be reused under A's key.

    Run 1 passes preflight at A; during norm stats the data becomes B and B's
    stats are written. The re-check refuses run 1. Run 2, on a restored A
    snapshot, must recompute rather than hit B's stats cached as A.
    """
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    launched: list[str] = []
    norm_runs: list[int] = []

    async def _swap_then_write_b_stats(ctx: TaskContext, spec: Any) -> int:
        launched.append(Path(spec.script_path).name)
        if launched[-1] == "compute_norm_stats.py" and not norm_runs:
            norm_runs.append(1)
            _hub_metadata(root, _EPISODE, SHA_B)
            _write(Path(spec.cwd), f"assets/{_CFG}/my_dataset/norm_stats.json", "B")
        elif launched[-1] == "compute_norm_stats.py":
            _write(Path(spec.cwd), f"assets/{_CFG}/my_dataset/norm_stats.json", "A")
        else:
            (Path(spec.cwd) / "checkpoints" / _CFG / "exp" / "100").mkdir(parents=True)
        return 0

    monkeypatch.setattr(pi05, "run_training_subprocess", _swap_then_write_b_stats)
    with pytest.raises(DatasetRevisionError):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert not _cached_stats(tmp_path, f"{_CFG}@{SHA_A}").exists()

    _hub_metadata(root, _EPISODE, SHA_A)  # a clean A snapshot is restored
    launched.clear()
    ctx = _context(tmp_path, _local_ref(root, SHA_A))
    ctx.output_dir = tmp_path / "out2"
    result = _run(ctx)
    assert launched == ["compute_norm_stats.py", "train.py"]  # recomputed, no stale hit
    assert result["dataset_revision_verified"] is True
    assert _cached_stats(tmp_path, f"{_CFG}@{SHA_A}").read_text() == "A"


def test_failed_norm_stats_publish_nothing(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _local_download(tmp_path / "my_dataset", SHA_A)

    async def _writes_then_fails(ctx: TaskContext, spec: Any) -> int:
        _write(Path(spec.cwd), f"assets/{_CFG}/my_dataset/norm_stats.json", "partial")
        return 1

    monkeypatch.setattr(pi05, "run_training_subprocess", _writes_then_fails)
    with pytest.raises(RuntimeError, match="compute_norm_stats exited with code 1"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert not _cached_stats(tmp_path, f"{_CFG}@{SHA_A}").exists()


def test_publish_keeps_an_entry_another_run_published_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent writers: the first published entry wins and is never removed."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cache = pi05._norm_stats_cache_root(_CFG, SHA_A)
    for run, stats in (("r1", "first"), ("r2", "second")):
        _write(tmp_path / run, f"assets/{_CFG}/my_dataset/norm_stats.json", stats)
    assert pi05._publish_norm_stats(tmp_path / "r1", cache, _CFG, "my_dataset") is True
    assert pi05._publish_norm_stats(tmp_path / "r2", cache, _CFG, "my_dataset") is False
    assert (cache / _CFG / "my_dataset" / "norm_stats.json").read_text() == "first"
    assert [p.name for p in (cache / _CFG).iterdir()] == ["my_dataset"]  # no staging left


def _gr00t_or_openvla_context(tmp_path: Path, runner: str, revision: str) -> TaskContext:
    ctx = _context(tmp_path, _local_ref(tmp_path / "d", revision))
    spec = ctx.task.spec
    assert isinstance(spec, TrainingTask)
    ctx.task.spec = spec.model_copy(update={"config": {"runner": runner}})
    return ctx


@pytest.mark.parametrize("revision", [SHA_A, "main"])
@pytest.mark.parametrize(
    "runner_cls, label",
    [(GR00TRunner, "GR00T"), (OpenVLARunner, "OpenVLA")],
)
def test_runners_without_verification_refuse_a_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_cls: Any, label: str,
    revision: str,
) -> None:
    """Regression: a pin these runners can't enforce is refused, not silently ignored."""

    async def _never(*_: Any, **__: Any) -> int:
        raise AssertionError("must refuse before launching anything")

    monkeypatch.setattr(gr00t_train, "run_training_subprocess", _never)
    monkeypatch.setattr(openvla_train, "run_training_subprocess", _never)
    ctx = _gr00t_or_openvla_context(tmp_path, label.lower(), revision)
    with pytest.raises(DatasetRevisionError, match=f"the {label} runner does not verify"):
        asyncio.run(runner_cls().run(ctx))


def test_runner_rejects_wrong_local_revision(tmp_path: Path, fake_openpi: list[Any]) -> None:
    root = _local_download(tmp_path / "my_dataset", SHA_B)
    with pytest.raises(DatasetRevisionError, match=f"downloaded from commit {SHA_B}"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert fake_openpi == []  # nothing launched


def test_runner_rejects_mixed_revision_dataset(
    tmp_path: Path, fake_openpi: list[Any]
) -> None:
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    _hub_metadata(root, _EPISODE, SHA_B)
    with pytest.raises(DatasetRevisionError, match=r"episode_000000\.parquet"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert fake_openpi == []


@pytest.mark.parametrize("rel", ["meta/tasks.jsonl", _STATS_FILE["v2.1"]])
def test_runner_stops_on_incomplete_snapshot_before_norm_stats(
    tmp_path: Path, fake_openpi: list[Any], rel: str
) -> None:
    """Regression: missing loader metadata stops the run before normalization or training.

    Neither norm-stats subprocess nor train.py starts, and the shared
    revision-keyed norm-stats cache is never linked or created.
    """
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    _remove(root, rel)
    with pytest.raises(DatasetRevisionError, match=rf"{re.escape(rel)}: missing"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert fake_openpi == []
    assert not (tmp_path / "out" / "assets").exists()
    assert not (tmp_path / "home" / ".odyssey" / "pi05_assets").exists()


def test_runner_rejects_unverifiable_sha_pin(tmp_path: Path, fake_openpi: list[Any]) -> None:
    root = _local_download(tmp_path / "my_dataset", None)
    with pytest.raises(DatasetRevisionError, match="no Hub download metadata"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert fake_openpi == []


def test_runner_pin_rejects_repo_id_naming_another_directory(
    tmp_path: Path, fake_openpi: list[Any]
) -> None:
    """Regression: dataset.ref is verified at A, but data.repo_id loads a sibling at B.

    openpi reads HF_LEROBOT_HOME / data.repo_id. With only a revision pinned
    (no timing declared) the #102 probe still runs, and its declared-vs-loaded
    check refuses the mission before the pin is trusted.
    """
    declared = _local_download(tmp_path / "data" / "pinned", SHA_A)
    _local_download(tmp_path / "data" / "actual", SHA_B)
    ctx = _context(tmp_path, _local_ref(declared, SHA_A), data={"repo_id": "actual"})
    with pytest.raises(DatasetMismatchError, match="actual"):
        _run(ctx)
    assert fake_openpi == []


def test_runner_pin_follows_repo_id_naming_the_declared_directory(
    tmp_path: Path, fake_openpi: list[Any]
) -> None:
    root = _local_download(tmp_path / "data" / "pinned", SHA_A)
    ctx = _context(tmp_path, _local_ref(root, SHA_A), data={"repo_id": "pinned"})
    assert _run(ctx)["dataset_revision_verified"] is True
    assert _scripts(fake_openpi) == ["openpi_bootstrap.py", "train.py"]


def test_runner_pin_fails_when_probe_cannot_read_config(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Revision-only pin + unreadable openpi config: a revision error, not a timing one."""

    async def _fails(*_: Any, **__: Any) -> OpenpiTrainingConfig:
        raise pi05.OpenpiProbeError("could not read openpi config 'cfg'")

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _fails)
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    with pytest.raises(DatasetRevisionError, match="could not read openpi config"):
        _run(_context(tmp_path, _local_ref(root, SHA_A)))
    assert fake_openpi == []


def test_runner_branch_pin_is_unverified_and_not_a_cache_key(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _never(*_: Any, **__: Any) -> OpenpiTrainingConfig:
        raise AssertionError("a branch pin needs no probe")

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _never)
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    result = _run(_context(tmp_path, _local_ref(root, "main")))
    assert result["dataset_revision"] == "main"
    assert result["dataset_revision_verified"] is False
    assert _cached_stats(tmp_path, _CFG).is_file()  # the shared, unkeyed entry


def test_runner_rejects_pinned_hub_dataset(tmp_path: Path, fake_openpi: list[Any]) -> None:
    """openpi can't load a hub dataset at a revision: refuse rather than train on HEAD."""
    ref = DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name", revision=SHA_A)
    with pytest.raises(DatasetRevisionError, match="use source: local"):
        _run(_context(tmp_path, ref))
    assert fake_openpi == []


def test_runner_unpinned_hub_dataset_still_runs(
    tmp_path: Path, fake_openpi: list[Any]
) -> None:
    ref = DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name")
    result = _run(_context(tmp_path, ref))
    assert result["dataset_revision"] is None
    assert result["dataset_revision_verified"] is False


# ---------------------------------------------------------------------------
# One resolution shared with #102's timing checks
# ---------------------------------------------------------------------------

def _count_probes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wrap the fake probe so each call is recorded."""
    calls: list[str] = []
    probe = pi05.probe_openpi_training_config

    async def _counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return await probe(*args, **kwargs)

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _counted)
    return calls


def _timed_context(tmp_path: Path, dataset: DatasetRef, **config: Any) -> TaskContext:
    ctx = _context(tmp_path, dataset, **config)
    spec = ctx.task.spec
    assert isinstance(spec, TrainingTask)
    ctx.task.spec = spec.model_copy(update={"control_hz": 10})
    return ctx


def test_timing_and_pin_share_one_probe_and_one_directory(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """control_hz (#102) and the revision pin read the SAME loaded directory."""
    calls = _count_probes(monkeypatch)
    root = _local_download(tmp_path / "my_dataset", SHA_A)  # info.json: 10 fps
    result = _run(_timed_context(tmp_path, _local_ref(root, SHA_A)))
    assert calls == [_CFG]  # one probe serves both checks
    assert result["dataset_revision_verified"] is True
    assert result["control_hz"] == 10


def test_timing_and_pin_reject_the_same_mismatch_once(
    tmp_path: Path, fake_openpi: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared != loaded fails once, with #102's error, before either check trusts it."""
    calls = _count_probes(monkeypatch)
    declared = _local_download(tmp_path / "data" / "pinned", SHA_A)
    _local_download(tmp_path / "data" / "actual", SHA_B)
    ctx = _timed_context(tmp_path, _local_ref(declared, SHA_A), data={"repo_id": "actual"})
    with pytest.raises(DatasetMismatchError, match="actual"):
        _run(ctx)
    assert calls == [_CFG]
    assert fake_openpi == []


def test_runner_pin_without_config_name_names_the_real_problem(
    tmp_path: Path, fake_openpi: list[Any]
) -> None:
    """Regression: a missing config_name is reported as such, not as a dataset error."""
    root = _local_download(tmp_path / "my_dataset", SHA_A)
    ctx = _context(tmp_path, _local_ref(root, SHA_A))
    spec = ctx.task.spec
    assert isinstance(spec, TrainingTask)
    ctx.task.spec = spec.model_copy(update={"config": {"runner": "pi05"}})
    with pytest.raises(RuntimeError, match=r"config\['config_name'\] is required"):
        _run(ctx)
    assert fake_openpi == []
