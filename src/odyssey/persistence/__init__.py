"""Persistence layer — Persistence ABC + built-in implementations."""

from odyssey.persistence.base import Persistence
from odyssey.persistence.in_memory import InMemoryPersistence

__all__ = ["InMemoryPersistence", "Persistence"]
