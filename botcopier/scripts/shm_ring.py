#!/usr/bin/env python3
"""Simple shared memory ring buffer used to pass events between processes.

The implementation is intentionally small and only supports a single
producer and single consumer.  Messages are stored as ``[length][type][data]``
where ``length`` is a 32bit little endian integer of the number of bytes
following (``type`` + ``data``).

The layout of the shared memory region matches the structure defined in
``shm_ring.h`` so that lightweight C/MQL producers can write directly to
it while Python readers consume with minimal copying.
"""
from __future__ import annotations

import mmap
import os
import struct
from typing import Optional, Tuple

MAGIC = 0x52494E47  # 'RING'
HEADER_FMT = "<IIII"  # magic, size, head, tail
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TRADE_MSG = 0
METRIC_MSG = 1


class ShmRing:
    """Shared memory backed ring buffer.

    The ring is a fixed size buffer with a small header containing the
    read (``head``) and write (``tail``) offsets.  The class exposes
    ``push`` and ``pop`` methods for the producer and consumer
    respectively.  Only a single producer and single consumer are
    supported.
    """

    def __init__(self, mm: mmap.mmap, size: int) -> None:
        self._mm = mm
        self.size = size

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, path: str, size: int) -> "ShmRing":
        """Create a new shared memory ring at ``path`` with ``size`` bytes."""
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        total = HEADER_SIZE + size
        os.ftruncate(fd, total)
        mm = mmap.mmap(fd, total)
        struct.pack_into(HEADER_FMT, mm, 0, MAGIC, size, 0, 0)
        return cls(mm, size)

    @classmethod
    def open(cls, path: str) -> "ShmRing":
        """Open an existing ring located at ``path``."""
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, 0)
        magic, size, _, _ = struct.unpack_from(HEADER_FMT, mm, 0)
        if magic != MAGIC:
            raise ValueError("invalid shm ring")
        return cls(mm, size)

    # ------------------------------------------------------------------
    # low level helpers
    # ------------------------------------------------------------------
    def _space(self, head: int, tail: int) -> int:
        if tail >= head:
            return self.size - (tail - head) - 1
        return head - tail - 1

    # ------------------------------------------------------------------
    # producer API
    # ------------------------------------------------------------------
    def push(self, msg_type: int, payload: bytes) -> bool:
        """Append ``payload`` with ``msg_type`` to the ring.

        Returns ``True`` on success, ``False`` if there was insufficient
        space.  The caller may retry later.
        """
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        data_len = len(payload) + 1  # include type byte
        needed = 4 + data_len
        magic, size, head, tail = struct.unpack_from(HEADER_FMT, self._mm, 0)
        free = self._space(head, tail)
        if needed > free:
            return False
        if tail + needed > size:
            # mark wrap
            struct.pack_into("<I", self._mm, HEADER_SIZE + tail, 0)
            tail = 0
        struct.pack_into("<I", self._mm, HEADER_SIZE + tail, data_len)
        tail += 4
        struct.pack_into("B", self._mm, HEADER_SIZE + tail, msg_type)
        self._mm[HEADER_SIZE + tail + 1 : HEADER_SIZE + tail + data_len] = payload
        tail = (tail + data_len) % size
        struct.pack_into("<II", self._mm, 8, head, tail)
        return True

    # ------------------------------------------------------------------
    # consumer API
    # ------------------------------------------------------------------
    def pop(self) -> Optional[Tuple[int, memoryview]]:
        """Retrieve the next message from the ring.

        Returns ``(msg_type, memoryview)`` or ``None`` when the ring is
        empty.
        """
        magic, size, head, tail = struct.unpack_from(HEADER_FMT, self._mm, 0)
        if head == tail:
            return None
        if head + 4 > size:
            head = 0
        length = struct.unpack_from("<I", self._mm, HEADER_SIZE + head)[0]
        if length == 0:  # wrapped
            head = 0
            length = struct.unpack_from("<I", self._mm, HEADER_SIZE + head)[0]
        start = head + 4
        msg_type = struct.unpack_from("B", self._mm, HEADER_SIZE + start)[0]
        payload = memoryview(self._mm)[HEADER_SIZE + start + 1 : HEADER_SIZE + start + length]
        head = (start + length) % size
        struct.pack_into("<II", self._mm, 8, head, tail)
        return msg_type, payload

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._mm.close()


__all__ = ["ShmRing", "TRADE_MSG", "METRIC_MSG"]
