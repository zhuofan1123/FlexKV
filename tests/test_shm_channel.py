"""Round-trip tests for the CE↔TE ShmChannel.

Run with::

    python3 -m pytest tests/test_shm_channel.py -v

The test forks N producer processes and one consumer; each producer sends K
graph-shaped messages and reads back acks via its own result ring.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

from flexkv.transfer.shm_channel import ShmChannel, ShmControlBlock


SERVER_ID = "shm_channel_test"


def _producer(channel_id: int, n: int, server_id: str) -> None:
    ch = ShmChannel(server_id, channel_id, create=False)
    for i in range(n):
        ch.submit_send({"channel": channel_id, "seq": i, "data": b"x" * 1024})
    # Wait for echoed acks.
    received = 0
    while received < n:
        msgs = ch.result_recv(timeout_s=2.0)
        received += len(msgs)
        for m in msgs:
            assert m["channel"] == channel_id


def _consumer(num_channels: int, total_per_channel: int, server_id: str) -> None:
    ctrl = ShmControlBlock(server_id, create=True)
    channels = [
        ShmChannel(server_id, i, create=True) for i in range(num_channels)
    ]
    ctrl.set_ready()
    pending = num_channels * total_per_channel
    while pending > 0:
        had_work = False
        for ch in channels:
            msgs = ch.submit_recv()
            if msgs:
                had_work = True
                for m in msgs:
                    ch.result_send({"channel": m["channel"], "seq": m["seq"]})
                    pending -= 1
        if not had_work:
            time.sleep(0.001)
    for ch in channels:
        ch.close()
        ch.unlink()
    ctrl.close()
    ctrl.unlink()


def _cleanup_shm(server_id: str, num_channels: int) -> None:
    for ch_id in range(num_channels):
        path = f"/dev/shm/flexkv_te_ch_{server_id}_{ch_id}"
        if os.path.exists(path):
            os.unlink(path)
    ctrl_path = f"/dev/shm/flexkv_te_ctrl_{server_id}"
    if os.path.exists(ctrl_path):
        os.unlink(ctrl_path)


def test_n_producer_one_consumer():
    server_id = f"{SERVER_ID}_npm"
    num_channels = 4
    total_per_channel = 32

    _cleanup_shm(server_id, num_channels)

    ctx = mp.get_context("spawn")
    consumer = ctx.Process(
        target=_consumer,
        args=(num_channels, total_per_channel, server_id),
    )
    consumer.start()

    # Wait for consumer to set up shm files.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if os.path.exists(f"/dev/shm/flexkv_te_ctrl_{server_id}"):
            break
        time.sleep(0.01)
    else:
        consumer.terminate()
        pytest.fail("consumer never created control block")

    # Wait for ready flag via control block.
    ctrl = ShmControlBlock(server_id, create=False)
    assert ctrl.wait_ready(timeout_s=5.0)
    ctrl.close()

    producers = [
        ctx.Process(
            target=_producer, args=(i, total_per_channel, server_id)
        )
        for i in range(num_channels)
    ]
    for p in producers:
        p.start()
    for p in producers:
        p.join(timeout=10.0)
        assert p.exitcode == 0, f"producer {p.pid} exit code {p.exitcode}"

    consumer.join(timeout=10.0)
    assert consumer.exitcode == 0


def test_single_round_trip_local():
    server_id = f"{SERVER_ID}_local"
    _cleanup_shm(server_id, 1)
    ctrl = ShmControlBlock(server_id, create=True)
    ch = ShmChannel(server_id, 0, create=True)
    try:
        ch.submit_send("hello")
        ch.submit_send({"k": 42})
        msgs = ch.submit_recv()
        assert msgs == ["hello", {"k": 42}]

        ch.result_send([1, 2, 3])
        out = ch.result_recv(timeout_s=0.0)
        assert out == [[1, 2, 3]]
    finally:
        ch.close()
        ch.unlink()
        ctrl.close()
        ctrl.unlink()
