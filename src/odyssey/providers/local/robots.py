"""LocalRobotProvider — validates RobotSpec without external services.

Two paths:

  * ``embodiment`` — a well-known name in our local catalog. Catalog is
    intentionally small for v0.1.0-alpha; users with their own embodiment
    can supply a URDF instead, or extend ``KNOWN_EMBODIMENTS`` in their
    own provider subclass.
  * ``urdf`` — a path on disk. Must exist; not otherwise parsed.

A spec with only ``id`` set isn't ours — the registry routes those to the
Lovell-mode provider.
"""

from __future__ import annotations

from pathlib import Path

from odyssey.providers.base import ResolvedRobot, RobotProvider
from odyssey.spec.mission import RobotSpec

KNOWN_EMBODIMENTS: frozenset[str] = frozenset(
    {
        "franka_panda",
        "ur5e",
        "ur10e",
        "panda",
        "kinova_gen3",
        "unitree_go2",
        "unitree_h1",
        "tiago",
        "stretch3",
    }
)


class LocalRobotProvider(RobotProvider):
    """Resolves ``embodiment`` and ``urdf`` robot specs locally."""

    @property
    def name(self) -> str:
        return "local"

    async def resolve(self, spec: RobotSpec) -> ResolvedRobot:
        if spec.embodiment is not None:
            if spec.embodiment not in KNOWN_EMBODIMENTS:
                raise ValueError(
                    f"Unknown embodiment {spec.embodiment!r}. "
                    f"Known: {sorted(KNOWN_EMBODIMENTS)}. "
                    "Supply a URDF path instead, or register a custom "
                    "RobotProvider for this embodiment."
                )
            return ResolvedRobot(
                provider=self.name,
                name=spec.embodiment,
                embodiment=spec.embodiment,
            )

        if spec.urdf is not None:
            urdf_path = Path(spec.urdf)
            if not urdf_path.is_file():
                raise FileNotFoundError(
                    f"URDF not found: {urdf_path}"
                )
            return ResolvedRobot(
                provider=self.name,
                name=urdf_path.stem,
                urdf_path=str(urdf_path),
            )

        raise ValueError(
            "LocalRobotProvider requires embodiment or urdf to be set; "
            "id-only specs route to the Lovell provider."
        )
