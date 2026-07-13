"""UR5e / Robotiq 2F-85 drug-sorting cell: embodiment spec + scripted policy."""

from __future__ import annotations

from odyssey.embodiments.ur5e_drugsort import embodiment
from odyssey.embodiments.ur5e_drugsort.policy import (
    GRIP_CLOSE,
    GRIP_OPEN,
    PHASES,
    PickPlaceController,
    Waypoint,
)

__all__ = [
    "GRIP_CLOSE",
    "GRIP_OPEN",
    "PHASES",
    "PickPlaceController",
    "Waypoint",
    "embodiment",
]
