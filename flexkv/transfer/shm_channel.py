# SPDX-License-Identifier: Apache-2.0
"""
Shared-memory IPC channel for CacheEngine ↔ TransferEngine communication in
multi-DP FlexKV.

Adapted from PR #144 (commit 5e262ca, originally for DPClient ↔ KVServer).
Slimmed down for the CE↔TE use case:
  - Per-channel size is small (256 KB ring + 256 KB sync) because the only
    payloads are pickled TransferOpGraph (submit) and CompletedOp lists (wait).
  - Sync request/response slot is dropped — CE↔TE is fully fire-and-forget on
    both directions; submit is async, completions are pushed asynchronously.
  - Two SPSC ring buffers per channel: `submit` (CE→TE) and `result` (TE→CE).
    Each side futex-waits on its own counter when the ring is empty.

Layout per channel (one /dev/shm file per CE):
  [0..64)        submit_write_pos  (uint64, CE writes)
  [64..128)      submit_read_pos   (uint64, TE writes)
  [128..192)     submit_wake       (int32, CE bumps + futex_wake)
  [192..256)     result_write_pos  (uint64, TE writes)
  [256..320)     result_read_pos   (uint64, CE writes)
  [320..384)     result_wake       (int32, TE bumps + futex_wake)
  [384..ring_off) reserved
  [ring_off ..)  submit ring (slot * SUBMIT_SLOTS)
                 result ring (slot * RESULT_SLOTS)

A single `ShmControlBlock` (separate /dev/shm file) carries a global wake
counter that the TE polls when it has more than one channel attached, so it
can sleep idly without per-channel futex wait.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import pickle
import platform
import struct
from typing import Any, List, Optional

# ── Linux futex wrappers ────────────────────────────────────────────────

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# futex syscall number is arch-specific; pick at import time.
_MACHINE = platform.machine().lower()
if _MACHINE in ("x86_64", "amd64"):
    _SYS_FUTEX = 202
elif _MACHINE in ("aarch64", "arm64"):
    _SYS_FUTEX = 98
else:  # pragma: no cover
    raise RuntimeError(f"Unsupported machine for futex syscall: {_MACHINE}")
_FUTEX_WAIT = 0
_FUTEX_WAKE = 1


def _futex_wait(addr: int, expected: int, timeout_ns: Optional[int] = None) -> int:
    if timeout_ns is None:
        return _libc.syscall(
            _SYS_FUTEX, ctypes.c_void_p(addr),
            _FUTEX_WAIT, ctypes.c_int(expected),
            ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0),
        )
    # struct timespec
    ts = (ctypes.c_long * 2)(timeout_ns // 1_000_000_000,
                             timeout_ns % 1_000_000_000)
    return _libc.syscall(
        _SYS_FUTEX, ctypes.c_void_p(addr),
        _FUTEX_WAIT, ctypes.c_int(expected),
        ctypes.byref(ts), ctypes.c_void_p(0), ctypes.c_int(0),
    )


def _futex_wake(addr: int, count: int = 1) -> int:
    return _libc.syscall(
        _SYS_FUTEX, ctypes.c_void_p(addr),
        _FUTEX_WAKE, ctypes.c_int(count),
        ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0),
    )


# ── Layout constants ────────────────────────────────────────────────────

_CL = 64  # cache line

# Header lives in the first 6 cache lines; ring data starts on a page boundary.
OFF_SUBMIT_W = 0 * _CL
OFF_SUBMIT_R = 1 * _CL
OFF_SUBMIT_WAKE = 2 * _CL
OFF_RESULT_W = 3 * _CL
OFF_RESULT_R = 4 * _CL
OFF_RESULT_WAKE = 5 * _CL
HEADER_SIZE = 6 * _CL  # 384 B

# Default sizing — overridable via constructor for tests.
DEFAULT_SUBMIT_SLOTS = 256          # power of 2
DEFAULT_RESULT_SLOTS = 256          # power of 2
DEFAULT_SLOT_SIZE = 64 * 1024       # 64 KB; transfer graphs are small

_PAGE = 4096


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# Length-prefix struct: 4-byte LE uint32 prefix in front of each pickle blob.
_LEN_HDR = struct.Struct("<I")


# ── ShmControlBlock ─────────────────────────────────────────────────────

CTRL_WAKE = 0
CTRL_READY = _CL
CTRL_SIZE = _PAGE


def _safe_id(server_id: str) -> str:
    return server_id.replace("/", "_").replace(":", "_").strip("_")


class ShmControlBlock:
    """Optional global wake counter used by the TE when polling N channels."""

    def __init__(self, server_id: str, create: bool = False):
        self.server_id = server_id
        safe = _safe_id(server_id)
        self.shm_path = f"/dev/shm/flexkv_te_ctrl_{safe}"

        if create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, CTRL_SIZE)
            self.buf = mmap.mmap(fd, CTRL_SIZE)
            os.close(fd)
            self.buf[:] = b"\x00" * CTRL_SIZE
        else:
            fd = os.open(self.shm_path, os.O_RDWR)
            self.buf = mmap.mmap(fd, CTRL_SIZE)
            os.close(fd)

        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self._wake = ctypes.c_int32.from_address(self._base + CTRL_WAKE)
        self._ready = ctypes.c_int32.from_address(self._base + CTRL_READY)

    @property
    def _wake_addr(self) -> int:
        return self._base + CTRL_WAKE

    @property
    def _ready_addr(self) -> int:
        return self._base + CTRL_READY

    def notify(self) -> None:
        # Read-modify-write is not atomic across processes; the TE uses snapshot
        # comparison so any change wakes it.
        self._wake.value += 1
        _futex_wake(self._wake_addr, 1)

    def get_wake(self) -> int:
        return self._wake.value

    def wait(self, expected: int, timeout_ns: Optional[int] = None) -> None:
        _futex_wait(self._wake_addr, expected, timeout_ns)

    def set_ready(self) -> None:
        self._ready.value = 1
        _futex_wake(self._ready_addr, 0x7FFFFFFF)

    def wait_ready(self, timeout_s: float = 60.0) -> bool:
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ready.value != 0:
                return True
            _futex_wait(self._ready_addr, 0,
                        timeout_ns=int(0.5 * 1_000_000_000))
        return False

    def close(self) -> None:
        if self.buf is not None:
            self.buf.close()
            self.buf = None

    def unlink(self) -> None:
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass


# ── ShmChannel (CE ↔ TE) ────────────────────────────────────────────────

class ShmChannel:
    """One bi-directional channel between a single CE and the TE.

    Two SPSC rings: CE→TE submit, TE→CE result. Each side futex-waits on its
    own wake counter when its consumer ring is empty.
    """

    def __init__(self,
                 server_id: str,
                 channel_id: int,
                 create: bool = False,
                 submit_slots: int = DEFAULT_SUBMIT_SLOTS,
                 result_slots: int = DEFAULT_RESULT_SLOTS,
                 slot_size: int = DEFAULT_SLOT_SIZE):
        assert submit_slots & (submit_slots - 1) == 0, \
            "submit_slots must be power of 2"
        assert result_slots & (result_slots - 1) == 0, \
            "result_slots must be power of 2"

        self.channel_id = channel_id
        self.submit_slots = submit_slots
        self.result_slots = result_slots
        self.slot_size = slot_size

        safe = _safe_id(server_id)
        self.shm_path = f"/dev/shm/flexkv_te_ch_{safe}_{channel_id}"

        # Lay out: header -> aligned to page -> submit ring -> result ring.
        self._submit_off = _round_up(HEADER_SIZE, _PAGE)
        self._result_off = self._submit_off + submit_slots * slot_size
        total = self._result_off + result_slots * slot_size

        self.total_size = total

        if create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, total)
            self.buf = mmap.mmap(fd, total)
            os.close(fd)
            self.buf[:HEADER_SIZE] = b"\x00" * HEADER_SIZE
        else:
            fd = os.open(self.shm_path, os.O_RDWR)
            self.buf = mmap.mmap(fd, total)
            os.close(fd)

        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self._submit_w = ctypes.c_uint64.from_address(self._base + OFF_SUBMIT_W)
        self._submit_r = ctypes.c_uint64.from_address(self._base + OFF_SUBMIT_R)
        self._submit_wake = ctypes.c_int32.from_address(self._base + OFF_SUBMIT_WAKE)
        self._result_w = ctypes.c_uint64.from_address(self._base + OFF_RESULT_W)
        self._result_r = ctypes.c_uint64.from_address(self._base + OFF_RESULT_R)
        self._result_wake = ctypes.c_int32.from_address(self._base + OFF_RESULT_WAKE)

    # ---- futex helpers ----

    @property
    def _submit_wake_addr(self) -> int:
        return self._base + OFF_SUBMIT_WAKE

    @property
    def _result_wake_addr(self) -> int:
        return self._base + OFF_RESULT_WAKE

    def _bump_wake(self, ptr: ctypes.c_int32, addr: int) -> None:
        ptr.value += 1
        _futex_wake(addr, 1)

    # ---- ring helpers ----

    def _ring_full(self, w: int, r: int, slots: int) -> bool:
        return ((w + 1) & (slots - 1)) == r

    def _write_blob(self, off: int, blob: bytes) -> None:
        n = len(blob)
        if 4 + n > self.slot_size:
            raise ValueError(
                f"shm channel slot too small for payload "
                f"({4 + n} > {self.slot_size}); raise slot_size"
            )
        self.buf[off:off + 4] = _LEN_HDR.pack(n)
        self.buf[off + 4:off + 4 + n] = blob

    def _read_blob(self, off: int) -> bytes:
        (n,) = _LEN_HDR.unpack_from(self.buf, off)
        return bytes(self.buf[off + 4:off + 4 + n])

    # ---- CE side: submit + recv result ----

    def submit_send(self, payload: Any) -> None:
        """Enqueue a payload to TE. Spins+yields if ring is full."""
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        wp = self._submit_w.value
        slots = self.submit_slots
        for spin in range(1_000_000):
            rp = self._submit_r.value
            if not self._ring_full(wp, rp, slots):
                break
            if spin > 1000:
                # Wait for TE to consume; it will bump our wake counter when it
                # advances submit_r. (We watch the consumer's progress
                # indirectly by polling submit_r.)
                os.sched_yield()
        else:
            raise RuntimeError("shm channel submit ring full")

        self._write_blob(self._submit_off + wp * self.slot_size, blob)
        self._submit_w.value = (wp + 1) & (slots - 1)
        self._bump_wake(self._submit_wake, self._submit_wake_addr)

    def result_recv(self, timeout_s: Optional[float] = None) -> List[Any]:
        """Drain pending TE→CE results. Blocks up to `timeout_s` if empty."""
        out: List[Any] = []
        rp = self._result_r.value
        wp = self._result_w.value
        slots = self.result_slots
        if rp == wp and timeout_s is not None and timeout_s > 0:
            wake = self._result_wake.value
            # Re-check; TE might have arrived between read and wait.
            wp = self._result_w.value
            if rp == wp:
                if timeout_s == float("inf"):
                    _futex_wait(self._result_wake_addr, wake)
                else:
                    _futex_wait(self._result_wake_addr, wake,
                                timeout_ns=int(timeout_s * 1_000_000_000))
                wp = self._result_w.value

        while rp != wp:
            blob = self._read_blob(self._result_off + rp * self.slot_size)
            out.append(pickle.loads(blob))
            rp = (rp + 1) & (slots - 1)
        if out:
            self._result_r.value = rp
        return out

    # ---- TE side: recv submit + send result ----

    def submit_recv(self) -> List[Any]:
        """Drain CE→TE submissions (non-blocking)."""
        out: List[Any] = []
        rp = self._submit_r.value
        wp = self._submit_w.value
        slots = self.submit_slots
        while rp != wp:
            blob = self._read_blob(self._submit_off + rp * self.slot_size)
            out.append(pickle.loads(blob))
            rp = (rp + 1) & (slots - 1)
        if out:
            self._submit_r.value = rp
        return out

    def result_send(self, payload: Any) -> None:
        """Enqueue a result to CE."""
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        wp = self._result_w.value
        slots = self.result_slots
        for spin in range(1_000_000):
            rp = self._result_r.value
            if not self._ring_full(wp, rp, slots):
                break
            if spin > 1000:
                os.sched_yield()
        else:
            raise RuntimeError("shm channel result ring full")

        self._write_blob(self._result_off + wp * self.slot_size, blob)
        self._result_w.value = (wp + 1) & (slots - 1)
        self._bump_wake(self._result_wake, self._result_wake_addr)

    # ---- Submit-wake fileno: lets TE selector wait on this channel ----

    @property
    def submit_wake_fd(self) -> int:
        # We don't have a real eventfd; selector users should poll get_wake()
        # delta + futex_wait via ShmControlBlock. Returning -1 signals "no fd".
        return -1

    # ---- Lifecycle ----

    def close(self) -> None:
        if self.buf is not None:
            self.buf.close()
            self.buf = None

    def unlink(self) -> None:
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
