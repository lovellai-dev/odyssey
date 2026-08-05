"""LocalRobotProvider — validates RobotSpec without external services.

Two paths:

  * ``embodiment`` — a well-known name in our local catalog. Catalog is
    intentionally small for v0.1.0-alpha; users with their own embodiment
    can supply a URDF instead, or extend ``KNOWN_EMBODIMENTS`` in their
    own provider subclass.
  * ``urdf`` — a path on disk. Must exist; not otherwise parsed.

A spec with only ``id`` set isn't ours — the registry routes those to the
Lovell-mode provider.

Scope note: this provider resolves the embodiment layer. In the Lovell
architecture a robot also has a loadout of agents (PILOT + SPECIALISTs)
that v0.0.x doesn't yet model in ``RobotSpec``, so there's no agent
layer for the provider to walk into here. When loadout-aware missions
ship, the local provider (for inline-declared loadouts) or a sibling
Lovell-mode provider (for hosted loadouts), or both, will extend to
resolve them — that decision hasn't been made yet.
"""

from __future__ import annotations

from pathlib import Path

from odyssey.providers.base import ResolvedRobot, RobotProvider
from odyssey.spec.mission import RobotSpec

# Scoped to embodiments at least one shipped runner can actually drive
# end-to-end. Two runners qualify a name today:
#   * Robosuite — the arms its built-in robot models cover
#     (``ROBOSUITE_ROBOT_NAMES`` in runners/evals/robosuite.py is the matching
#     translation table).
#   * the GR00T runner — arms it can finetune as a NEW_EMBODIMENT (e.g.
#     ``ur10e``), which Robosuite doesn't ship and so stays outside
#     ``ROBOSUITE_ROBOT_NAMES``.
# The catalog is therefore a superset of the Robosuite-drivable names, not a
# mirror of them. Quadrupeds (unitree_go2/h1) and mobile bases (tiago,
# stretch3) remain excluded — with no runner that honors them they accepted
# with a green spec, then produced identical default-Panda eval runs that
# misled users about what was being simulated. Re-add a name here only when a
# runner exists that drives it; for unsupported robots, ``urdf:`` still works.
KNOWN_EMBODIMENTS: frozenset[str] = frozenset(
    {
        "franka_panda",   # alias of "panda" — common in OpenVLA / LeRobot specs
        "panda",
        "sawyer",
        "iiwa",
        "jaco",
        "kinova_gen3",
        "ur5e",
        "ur10e",          # driven by the GR00T runner (NEW_EMBODIMENT finetune),
                          # not Robosuite — hence outside ROBOSUITE_ROBOT_NAMES.
        "baxter",
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
