"""Unit tests for ChunkPilotAdapter — buffer drain, re-query, and chunk-aware flush.

CPU-only: a fake policy client + trivial obs_builder/action_mapper exercise the
buffering logic without any model, GPU, or GR00T dependency.
"""

from __future__ import annotations

from typing import Any

from odyssey.runners.evals.chunk_pilot_adapter import ChunkPilotAdapter


class FakeClient:
    """Records get_action calls; returns a fresh chunk each time.

    A chunk is ``{"actions": [f"{tag}-0", ...], "obs": server_obs}`` so tests can
    assert which server observation produced which chunk.
    """

    def __init__(self, n: int, *, as_tuple: bool = False) -> None:
        self._n = n
        self._as_tuple = as_tuple
        self.calls: list[Any] = []  # the server_obs of each get_action call

    def get_action(self, server_obs: Any) -> Any:
        idx = len(self.calls)
        self.calls.append(server_obs)
        chunk = {"actions": [f"q{idx}-a{k}" for k in range(self._n)], "obs": server_obs}
        return (chunk, {"info": idx}) if self._as_tuple else chunk


def _obs_builder(observation: Any, instruction: str) -> dict[str, Any]:
    return {"obs": observation, "instr": instruction}


def _action_mapper(chunk: Any, k: int) -> Any:
    return chunk["actions"][k]


def _make(n: int = 4, *, as_tuple: bool = False) -> tuple[ChunkPilotAdapter, FakeClient]:
    client = FakeClient(n, as_tuple=as_tuple)
    adapter = ChunkPilotAdapter(
        client, obs_builder=_obs_builder, action_mapper=_action_mapper, n_action_steps=n
    )
    return adapter, client


def test_satisfies_pilot_runtime_protocol() -> None:
    from odyssey.runners.agents.runtime import PilotRuntime

    adapter, _ = _make()
    assert isinstance(adapter, PilotRuntime)  # runtime_checkable: has act()


def test_one_query_per_drained_chunk() -> None:
    adapter, client = _make(n=4)
    adapter.set_obs({"eef": 1})
    actions = [adapter.act(image=None, instruction="pick") for _ in range(4)]
    assert actions == ["q0-a0", "q0-a1", "q0-a2", "q0-a3"]
    assert len(client.calls) == 1  # a whole chunk drained from a single query


def test_requery_after_exhaustion() -> None:
    adapter, client = _make(n=4)
    adapter.set_obs({"eef": 1})
    for _ in range(4):
        adapter.act(image=None, instruction="pick")
    assert len(client.calls) == 1
    fifth = adapter.act(image=None, instruction="pick")
    assert len(client.calls) == 2  # 5th step re-queries
    assert fifth == "q1-a0"  # cursor reset to 0 of the new chunk


def test_flush_on_instruction_change() -> None:
    adapter, client = _make(n=4)
    adapter.set_obs({"eef": 1})
    a0 = adapter.act(image=None, instruction="pick up the mug")
    assert a0 == "q0-a0" and len(client.calls) == 1
    # instruction changes mid-chunk (a phase advance) -> discard stale chunk, re-query
    b0 = adapter.act(image=None, instruction="place it in the basket")
    assert len(client.calls) == 2
    assert b0 == "q1-a0"  # fresh chunk, cursor reset
    # the re-query used the NEW instruction
    assert client.calls[1]["instr"] == "place it in the basket"


def test_same_instruction_keeps_draining_even_if_obs_changes() -> None:
    # within a phase the chunk is open-loop: obs updates but the buffered chunk drains
    adapter, client = _make(n=3)
    adapter.set_obs({"step": 0})
    adapter.act(image=None, instruction="pick")
    adapter.set_obs({"step": 1})
    adapter.act(image=None, instruction="pick")
    assert len(client.calls) == 1  # no re-query mid-chunk


def test_set_obs_threaded_to_builder() -> None:
    adapter, client = _make(n=2)
    adapter.set_obs({"eef_pos": [0.1, 0.2, 0.3]})
    adapter.act(image=None, instruction="grasp")
    # the server obs handed to the client came from obs_builder(set_obs value, instr)
    assert client.calls[0] == {"obs": {"eef_pos": [0.1, 0.2, 0.3]}, "instr": "grasp"}


def test_reset_flushes_buffer() -> None:
    adapter, client = _make(n=4)
    adapter.set_obs({"eef": 1})
    adapter.act(image=None, instruction="pick")
    assert len(client.calls) == 1
    adapter.reset()
    adapter.act(image=None, instruction="pick")  # same instruction, but buffer flushed
    assert len(client.calls) == 2


def test_tuple_result_is_unwrapped() -> None:
    adapter, _client = _make(n=2, as_tuple=True)
    adapter.set_obs({"eef": 1})
    a0 = adapter.act(image=None, instruction="pick")
    assert a0 == "q0-a0"  # used chunk[0] of the (chunk, info) tuple
