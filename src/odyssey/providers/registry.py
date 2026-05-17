"""ProviderRegistry — routes specs to providers without the engine knowing.

Routing rules:

  * RobotSpec with ``id`` set → registered provider for ``"lovell"`` (or any
    other ``id``-handling provider). With ``urdf`` or ``embodiment`` set →
    the registered ``"local"`` provider.
  * ModelRef → provider registered for its ``source`` (huggingface /
    from_task / lovell).
  * DatasetRef → provider whose ``supported_sources`` contains the ref's
    ``source``.

Raises ``ProviderNotRegisteredError`` when no provider matches. The error
message names what was requested so the user sees ``no model provider for
source='huggingface'`` instead of a KeyError.
"""

from __future__ import annotations

from odyssey.providers.base import DatasetProvider, ModelProvider, RobotProvider
from odyssey.spec.mission import RobotSpec
from odyssey.spec.refs import DatasetRef, DatasetSource, ModelRef


class ProviderNotRegisteredError(Exception):
    """Raised when a spec needs a provider that isn't registered."""


class ProviderRegistry:
    def __init__(self) -> None:
        self._robot_by_kind: dict[str, RobotProvider] = {}
        self._model_by_source: dict[str, ModelProvider] = {}
        self._dataset_by_source: dict[DatasetSource, DatasetProvider] = {}

    # ---- registration ----

    def register_robot(self, provider: RobotProvider, *, handles: str) -> None:
        """Register `provider` as the handler for robot specs of `handles`
        kind (``"local"`` for embodiment/urdf specs, ``"lovell"`` for id
        specs)."""
        self._robot_by_kind[handles] = provider

    def register_model(self, provider: ModelProvider) -> None:
        self._model_by_source[provider.source] = provider

    def register_dataset(self, provider: DatasetProvider) -> None:
        for source in provider.supported_sources:
            self._dataset_by_source[source] = provider

    # ---- routing ----

    def for_robot_spec(self, spec: RobotSpec) -> RobotProvider:
        kind = "lovell" if spec.id is not None else "local"
        try:
            return self._robot_by_kind[kind]
        except KeyError as e:
            raise ProviderNotRegisteredError(
                f"no robot provider registered for kind={kind!r}"
            ) from e

    def for_model_ref(self, ref: ModelRef) -> ModelProvider:
        source = ref.source
        try:
            return self._model_by_source[source]
        except KeyError as e:
            raise ProviderNotRegisteredError(
                f"no model provider registered for source={source!r}"
            ) from e

    def for_dataset_ref(self, ref: DatasetRef) -> DatasetProvider:
        try:
            return self._dataset_by_source[ref.source]
        except KeyError as e:
            raise ProviderNotRegisteredError(
                f"no dataset provider registered for source={ref.source.value!r}"
            ) from e
