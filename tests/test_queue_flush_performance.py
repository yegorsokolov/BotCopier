import time


def _flush_shift(queue: list[str]) -> None:
    i = 0
    while i < len(queue):
        queue.pop(i)


def _flush_index(queue: list[str]) -> None:
    head = 0
    tail = len(queue)
    while head < tail:
        head += 1
    del queue[:head]


def test_flush_performance() -> None:
    backlog = [str(i) for i in range(20000)]

    q1 = backlog.copy()
    start = time.perf_counter()
    _flush_shift(q1)
    shifted = time.perf_counter() - start

    q2 = backlog.copy()
    start = time.perf_counter()
    _flush_index(q2)
    indexed = time.perf_counter() - start

    assert shifted > indexed * 5
