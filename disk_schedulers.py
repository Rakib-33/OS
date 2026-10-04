"""Batch disk schedulers. Cylinders are numbered 0 through 199."""

from __future__ import annotations

from dataclasses import dataclass

MAX_CYLINDER = 199
START_HEAD = 100
SCHEDULERS = ("FCFS", "SCAN", "C-SCAN", "SSTF")


@dataclass(frozen=True)
class Schedule:
    name: str
    path: tuple[int, ...]
    seek: int


def schedule(name: str, requests: list[int] | tuple[int, ...], head: int = START_HEAD,
             direction: int = 1) -> Schedule:
    """Schedule one fully queued batch; direction is +1 (up) or -1 (down).

    SCAN touches the physical endpoint only when a reversal is needed.
    C-SCAN touches the endpoint and counts the full wrap only when requests
    exist on the other side of the head. Endpoints are not request services.
    Equal-distance SSTF requests use the lower cylinder, then arrival order.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler: {name}")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    if not 0 <= head <= MAX_CYLINDER:
        raise ValueError("head outside disk")
    if any(not isinstance(r, int) or not 0 <= r <= MAX_CYLINDER for r in requests):
        raise ValueError("requests must be integer cylinders 0..199")
    req = list(requests)
    path = [head]
    if name == "FCFS":
        path.extend(req)
    elif name == "SSTF":
        remaining = list(enumerate(req))
        while remaining:
            choice = min(remaining, key=lambda item: (abs(item[1] - path[-1]), item[1], item[0]))
            _, target = choice
            path.append(target)
            remaining.remove(choice)
    else:
        ahead = sorted((r for r in req if (r - head) * direction >= 0), reverse=direction < 0)
        behind = sorted((r for r in req if (r - head) * direction < 0), reverse=direction > 0)
        path.extend(ahead)
        if behind:
            endpoint = MAX_CYLINDER if direction > 0 else 0
            other_end = 0 if direction > 0 else MAX_CYLINDER
            if path[-1] != endpoint:
                path.append(endpoint)
            if name == "C-SCAN":
                path.append(other_end)
                path.extend(reversed(behind))
            else:
                path.extend(behind)
    seek = sum(abs(b - a) for a, b in zip(path, path[1:]))
    return Schedule(name, tuple(path), seek)
