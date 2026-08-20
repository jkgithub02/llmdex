"""Attaching to a run that is already going.

Ingest returns the card immediately and leaves the agents running, so by the
time anyone opens that card the run is in flight. A reader arriving late must
get the whole trace, not the tail.
"""

import threading
import time

from app.core.events import AgentEvent
from app.features.agents.live import _DONE, Run


def test_a_late_reader_gets_everything_that_already_happened():
    run = Run("a/one", ["about"])
    run._publish(AgentEvent(agent="about", kind="phase", phase="started"))
    run._publish(AgentEvent(agent="about", kind="reasoning", text="thinking"))
    run._publish(_DONE)

    seen = list(run.follow())

    assert [e.kind for e in seen] == ["phase", "reasoning"]


def test_a_reader_follows_events_that_arrive_after_it_attaches():
    run = Run("a/one", ["about"])
    run._publish(AgentEvent(agent="about", kind="phase", phase="started"))

    seen: list[AgentEvent] = []

    def read() -> None:
        seen.extend(run.follow())

    reader = threading.Thread(target=read)
    reader.start()
    time.sleep(0.05)

    run._publish(AgentEvent(agent="about", kind="reasoning", text="later"))
    run._publish(_DONE)
    reader.join(timeout=2)

    assert [e.kind for e in seen] == ["phase", "reasoning"]
    assert seen[-1].text == "later"


def test_two_readers_each_get_the_whole_trace():
    run = Run("a/one", ["about"])
    run._publish(AgentEvent(agent="about", kind="phase", phase="started"))

    results: list[list[AgentEvent]] = [[], []]

    def read(slot: int) -> None:
        results[slot].extend(run.follow())

    readers = [threading.Thread(target=read, args=(i,)) for i in range(2)]
    for r in readers:
        r.start()
    time.sleep(0.05)
    run._publish(AgentEvent(agent="about", kind="phase", phase="done"))
    run._publish(_DONE)
    for r in readers:
        r.join(timeout=2)

    assert [len(r) for r in results] == [2, 2]


def test_a_finished_run_still_replays_and_then_stops():
    """A trace read after the run ended must terminate, not block forever."""
    run = Run("a/one", ["about"])
    run._publish(AgentEvent(agent="about", kind="phase", phase="done"))
    run._publish(_DONE)

    seen: list[AgentEvent] = []

    def read() -> None:
        seen.extend(run.follow())

    reader = threading.Thread(target=read)
    reader.start()
    reader.join(timeout=2)

    assert not reader.is_alive(), "follow() did not return on a finished run"
    assert len(seen) == 1


def test_a_subscriber_is_dropped_when_it_stops_reading():
    """A client that navigates away must not leave a queue growing forever."""
    run = Run("a/one", ["about"])
    run._publish(AgentEvent(agent="about", kind="phase", phase="started"))

    stream = run.follow()
    next(stream)  # drains the backlog, which is what attaches the subscriber
    stream.close()

    run._publish(AgentEvent(agent="about", kind="phase", phase="done"))

    assert run._subscribers == []
