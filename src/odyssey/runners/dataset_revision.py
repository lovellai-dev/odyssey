"""Dataset revision pinning: verify the data on disk is the declared version.

``DatasetRef.revision`` names the exact dataset version a task trains on. A
pin is only worth something if it is checked, so this module compares it
with what is actually on disk whenever that can be known:

  * A local copy made with ``hf download --local-dir`` carries Hub download
    metadata (``.cache/huggingface/download/<file>.metadata``) whose first
    line is the commit sha the file came from. ``meta/info.json`` is read as
    the dataset's marker file.
  * A revision that is not a sha (a branch or tag) can't be compared with a
    commit sha. It is recorded, not verified.
  * A local copy without download metadata is recorded, not verified.

A mismatch fails the task before any GPU work. Nothing here downloads.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from odyssey.spec.refs import DatasetRef, DatasetSource

logger = logging.getLogger(__name__)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_MARKER = Path("meta") / "info.json"
_DOWNLOAD_META = Path(".cache") / "huggingface" / "download"


class DatasetRevisionError(RuntimeError):
    """The data on disk is not the dataset revision the task declares."""


def is_commit_sha(revision: str) -> bool:
    """True for a full or abbreviated (>= 7 hex chars) commit sha."""
    return bool(_SHA_RE.match(revision.lower()))


def local_dataset_revision(root: str | Path) -> str | None:
    """Commit sha a local Hub download came from, or None if unknown.

    Reads the download metadata ``hf download --local-dir`` writes next to
    each file; its first line is the commit sha.
    """
    meta = Path(root) / _DOWNLOAD_META / f"{_MARKER}.metadata"
    try:
        first = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return first.lower() if is_commit_sha(first) else None


def check_local_dataset_revision(task_name: str, dataset: DatasetRef | None) -> None:
    """Fail when a local dataset's on-disk revision differs from the declared one.

    No-op without a declared revision or for non-local sources (those are the
    provider's or the runner's concern).
    """
    if dataset is None or dataset.revision is None:
        return
    if dataset.source != DatasetSource.LOCAL or not os.path.isabs(dataset.ref):
        return
    declared = dataset.revision.lower()
    if not is_commit_sha(declared):
        logger.info(
            "task %s: dataset revision %r is not a commit sha; recorded, not verified",
            task_name,
            dataset.revision,
        )
        return
    on_disk = local_dataset_revision(dataset.ref)
    if on_disk is None:
        logger.info(
            "task %s: dataset revision %s recorded but not verifiable: %s has no "
            "Hub download metadata",
            task_name,
            dataset.revision,
            dataset.ref,
        )
        return
    if not (on_disk.startswith(declared) or declared.startswith(on_disk)):
        raise DatasetRevisionError(
            f"task {task_name!r}: dataset revision {dataset.revision} declared, but "
            f"{dataset.ref} was downloaded from commit {on_disk}. Download the "
            "declared revision (hf download <repo> --repo-type dataset --revision "
            "<sha> --local-dir <dir>) or update the revision."
        )
