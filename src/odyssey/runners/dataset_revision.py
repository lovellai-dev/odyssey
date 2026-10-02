"""Dataset revision pinning: verify the data on disk is the declared version.

``DatasetRef.revision`` names the exact dataset version a task trains on. A
pin is only worth something if it is checked, so this module compares it
with EVERY file of a local copy, the way ``huggingface_hub`` itself decides
whether a local file is still the one it downloaded:

  * ``hf download --local-dir`` writes, per file, download metadata
    (``.cache/huggingface/download/<file>.metadata``): the commit sha the
    file came from, its etag and the download timestamp.
  * A file is verified when it has that metadata, its commit is the pinned
    one, and it hasn't been modified since (``st_mtime - 1 <= timestamp``,
    the Hub's own rule in ``huggingface_hub/_local_folder.py``).
  * Any file that fails — another revision, edited or replaced locally,
    added by hand — fails the whole dataset. One file's marker is never
    taken as proof of the rest: Hub metadata describes single files, not an
    atomic snapshot.
  * The copy must also be complete: every file the LeRobot v2.x manifest
    (``meta/info.json`` + ``meta/episodes.jsonl``) says training reads must
    be present, so a partial download can't pass on the files it has.

A commit-sha pin that can't be verified is an error, not a log line: the
pin claims an identity (it also keys the π0.5 norm-stats cache), so data
that may not match it must not train. A branch or tag is not an identity —
it moves — so it is recorded as unverified and nothing is keyed on it.

Nothing here downloads, and file contents are not hashed: the check costs
one ``stat`` and one small read per file, not a pass over the data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from odyssey.spec.refs import DatasetRef, DatasetSource

logger = logging.getLogger(__name__)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_CACHE_DIR = ".cache"
_DOWNLOAD_META = Path(_CACHE_DIR) / "huggingface" / "download"
# Problems listed in an error message before "… and N more".
_MAX_REPORTED = 5


class DatasetRevisionError(RuntimeError):
    """The data on disk is not (or can't be shown to be) the declared revision."""


def is_commit_sha(revision: str) -> bool:
    """True for a full or abbreviated (>= 7 hex chars) commit sha."""
    return bool(_SHA_RE.match(revision.lower()))


def _same_commit(on_disk: str, declared: str) -> bool:
    """Equal shas, allowing either one to be abbreviated (>= 7 hex chars).

    Both must be shas: an empty or truncated metadata line is a prefix of
    every sha and must never count as a match.
    """
    if not (is_commit_sha(on_disk) and is_commit_sha(declared)):
        return False
    return on_disk.startswith(declared) or declared.startswith(on_disk)


def _file_problem(root: Path, rel: Path, declared: str) -> str | None:
    """Why ``root/rel`` is not verifiably from commit ``declared``, or None."""
    meta = root / _DOWNLOAD_META / f"{rel}.metadata"
    try:
        lines = meta.read_text(encoding="utf-8").splitlines()
        commit, timestamp = lines[0].strip().lower(), float(lines[2].strip())
    except (OSError, IndexError, ValueError):
        return f"{rel}: no Hub download metadata (added or copied by hand?)"
    if not _same_commit(commit, declared):
        return f"{rel}: downloaded from commit {commit or '(none recorded)'}"
    try:
        mtime = (root / rel).stat().st_mtime
    except OSError:  # a broken symlink, or removed while we walked
        return f"{rel}: unreadable"
    if mtime - 1 > timestamp:
        return f"{rel}: modified after download"
    return None


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def dataset_files(root: Path) -> list[Path]:
    """Every file under ``root`` (relative), skipping the Hub's ``.cache``.

    A directory symlink pointing OUTSIDE the dataset is followed — parquet
    or video folders often live on another volume, and training reads
    through them. One pointing INSIDE it (an alias such as
    ``videos_latest -> videos``) is skipped: its files are verified at
    their real path, where the download metadata is. A symlink back to an
    ancestor (a cycle) is never re-entered.
    """
    files: list[Path] = []
    real_root = os.path.realpath(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        if here == root and _CACHE_DIR in dirnames:
            dirnames.remove(_CACHE_DIR)
        ancestors = {os.path.realpath(p) for p in [here, *here.parents]}
        for name in list(dirnames):
            link = here / name
            if not link.is_symlink():
                continue
            target = os.path.realpath(link)
            if _inside(target, real_root) or target in ancestors:
                dirnames.remove(name)
        files += [Path(dirpath, name).relative_to(root) for name in filenames]
    return sorted(files)


def _lerobot_expected_files(root: Path) -> list[Path]:
    """The files a LeRobot v2.x loader reads, from the dataset's own manifest.

    ``meta/info.json`` holds the ``data_path`` / ``video_path`` templates and
    ``chunks_size``; ``meta/episodes.jsonl`` lists the episode indices (else
    ``range(total_episodes)``). Raises ``DatasetRevisionError`` when that
    manifest is missing or not a v2.x layout — then completeness can't be
    established, and neither can the pin.
    """
    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        data_path, chunks = str(info["data_path"]), int(info["chunks_size"])
        video_path = info.get("video_path")
        features = info.get("features") or {}
        total = int(info["total_episodes"])
    except (OSError, ValueError, KeyError, TypeError):
        raise DatasetRevisionError(
            f"{root}: meta/info.json is missing or not a LeRobot v2.x manifest, so "
            "the files training reads (and whether all are present) are unknown"
        ) from None
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    try:
        lines = episodes_jsonl.read_text(encoding="utf-8").splitlines()
        episodes = [int(json.loads(line)["episode_index"]) for line in lines if line.strip()]
    except OSError:
        episodes = list(range(total))
    except (ValueError, KeyError, TypeError):
        raise DatasetRevisionError(f"{root}: meta/episodes.jsonl is malformed") from None
    video_keys = [
        key for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    ]
    expected = [Path("meta") / "info.json"]
    for ep in episodes:
        fields = {"episode_chunk": ep // chunks, "episode_index": ep}
        expected.append(Path(data_path.format(**fields)))
        if video_path:
            expected += [
                Path(str(video_path).format(video_key=key, **fields)) for key in video_keys
            ]
    return expected


def verify_local_dataset_revision(root: str | Path, declared: str) -> None:
    """Fail unless EVERY file under ``root`` is verifiably from commit ``declared``.

    Raises ``DatasetRevisionError`` listing the offending files: one from
    another revision, modified after download, with no download metadata at
    all, or one the LeRobot manifest says training reads but that isn't on
    disk. An empty or missing directory is unverifiable too.
    """
    root = Path(root)
    declared = declared.lower()
    files = dataset_files(root) if root.is_dir() else []
    if not files:
        raise DatasetRevisionError(f"{root} has no files to verify")
    # Present files must all be the pinned commit's; the files training reads
    # must all be present. A partial copy (an --include download, an
    # interrupted one) would otherwise pass, and the loader may fetch what is
    # missing from another revision.
    present = set(files)
    missing = [rel for rel in _lerobot_expected_files(root) if rel not in present]
    problems = [f"{rel}: missing (partial download?)" for rel in missing]
    problems += [p for rel in files if (p := _file_problem(root, rel, declared))]
    if problems:
        shown = "; ".join(problems[:_MAX_REPORTED])
        more = len(problems) - _MAX_REPORTED
        raise DatasetRevisionError(
            f"{root} is not verifiably commit {declared} "
            f"({len(problems)} problems, {len(files)} files on disk): {shown}"
            + (f"; … and {more} more" if more > 0 else "")
        )


def pins_commit(dataset: DatasetRef | None) -> bool:
    """True when ``dataset`` pins a commit sha (the only verifiable revision)."""
    return (
        dataset is not None
        and dataset.revision is not None
        and is_commit_sha(dataset.revision)
    )


def check_local_dataset_revision(
    task_name: str, dataset: DatasetRef | None, loaded_dir: str | None
) -> bool:
    """Verify the dataset training loads against the declared revision.

    ``loaded_dir`` is the directory training will read, as the runner
    resolved it (for π0.5: ``probe_openpi_training_config``), after it
    checked that it IS the declared dataset. Verifying it — not
    ``dataset.ref`` — keeps the pin about the data actually trained on.

    Returns True when the revision is a commit sha and every file matches
    it — the only case where the revision is a verified identity. Returns
    False without a declared revision, for a non-local source (the
    provider's or the runner's concern), or for a branch or tag (logged).
    Raises ``DatasetRevisionError`` when a pinned sha is contradicted by,
    or can't be verified against, the data training loads.
    """
    if dataset is None or dataset.revision is None:
        return False
    if dataset.source != DatasetSource.LOCAL:
        return False
    if not pins_commit(dataset):
        logger.info(
            "task %s: dataset revision %r is not a commit sha; recorded, not "
            "verified, and not used as a cache key",
            task_name,
            dataset.revision,
        )
        return False
    remedy = (
        "Download the pinned revision into a clean directory (hf download <repo> "
        "--repo-type dataset --revision <sha> --local-dir <dir>), point "
        "dataset.ref at it (an absolute path), copy it preserving mtimes "
        "(cp -p / rsync -a), or update the revision."
    )
    if not os.path.isabs(dataset.ref) or loaded_dir is None:
        raise DatasetRevisionError(
            f"task {task_name!r}: dataset revision {dataset.revision} is pinned, but "
            f"the directory training loads could not be resolved from {dataset.ref!r}"
            f" (needs an absolute dataset.ref and a LeRobot data.repo_id). {remedy}"
        )
    try:
        verify_local_dataset_revision(loaded_dir, dataset.revision)
    except DatasetRevisionError as exc:
        raise DatasetRevisionError(
            f"task {task_name!r}: dataset revision {dataset.revision} is pinned, but "
            f"{exc}. {remedy}"
        ) from None
    return True
