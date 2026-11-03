"""Bounded completion events for caller-owned, already-approved Popen handles.

This module never launches, retries, terminates or authorizes a scientific job.
Stdout must go to a durable file (or DEVNULL), not an undrained PIPE.
"""

import datetime
import math
import pathlib
import queue
import subprocess
import threading
import time
from dataclasses import dataclass


def _positive(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")


def _ssh_keepalive(argv):
    """Validate bounded SSH connection/keepalive settings; never invoke SSH."""
    if pathlib.Path(argv[0]).name.lower() not in ("ssh", "ssh.exe"):
        return
    options = {}
    for i, arg in enumerate(argv[:-1]):
        if arg == "-o" and "=" in argv[i + 1]:
            key, value = argv[i + 1].split("=", 1)
            if key.lower() in options:
                raise ValueError("duplicate SSH option: " + key)
            options[key.lower()] = value
    if options.get("batchmode", "").lower() != "yes":
        raise ValueError("SSH requires BatchMode=yes")
    for key in ("connecttimeout", "serveraliveinterval", "serveralivecountmax"):
        try:
            value = int(options[key])
        except (KeyError, ValueError):
            raise ValueError("SSH requires explicit positive " + key)
        if value <= 0:
            raise ValueError("SSH requires explicit positive " + key)


@dataclass(frozen=True)
class ForegroundJob:
    name: str
    process: subprocess.Popen
    approved_argv: tuple
    approval_reference: str
    terminal_locations: tuple
    read_terminal: object
    expected_terminal_identity: dict
    transport_seconds: float
    started_monotonic: float

    def validate(self):
        if not self.name or not self.approval_reference or not self.terminal_locations:
            raise ValueError("name, root approval reference and terminal locations required")
        if not isinstance(self.process.args, (list, tuple)):
            raise ValueError("explicit argv required; shell command strings rejected")
        if tuple(self.process.args) != tuple(self.approved_argv):
            raise ValueError("actual Popen argv differs from caller-approved argv")
        if not self.approved_argv or not all(isinstance(x, str) for x in self.approved_argv):
            raise ValueError("approved argv must contain strings")
        _ssh_keepalive(self.approved_argv)
        if self.process.stdout is not None or self.process.stderr is not None:
            raise ValueError("PIPE output rejected: use durable files or DEVNULL")
        if self.process.stdin is not None and not self.process.stdin.closed:
            raise ValueError("close/write-complete stdin before registering observation")
        if not callable(self.read_terminal) or not self.expected_terminal_identity:
            raise ValueError("bounded terminal reader and exact root identity required")
        _positive(self.transport_seconds, "transport_seconds")
        if not math.isfinite(self.started_monotonic) or self.started_monotonic > time.monotonic():
            raise ValueError("started_monotonic must be the actual launch clock, not future")


def _event(job, status, reason, returncode=None, terminal=None):
    return {
        "event": "FOREGROUND_JOB_TERMINAL",
        "job": job.name,
        "status": status,
        "reason": reason,
        "observed_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "local_pid": job.process.pid,
        "transport_returncode": returncode,
        "approval_reference": job.approval_reference,
        "approved_argv": list(job.approved_argv),
        "terminal_locations": list(job.terminal_locations),
        "terminal": terminal,
        "automatic_retry": False,
        "process_terminated_by_observer": False,
    }


def _assess(job, rc, terminal):
    if not isinstance(terminal, dict):
        return _event(job, "UNRESOLVED", "TERMINAL_RECORD_ABSENT", rc)
    if any(terminal.get(k) != v for k, v in job.expected_terminal_identity.items()):
        return _event(job, "UNRESOLVED", "TERMINAL_IDENTITY_MISMATCH", rc, terminal)
    if rc != 0:
        return _event(job, "UNRESOLVED", "TRANSPORT_NONZERO_REVIEW_SAVED_OS", rc, terminal)
    cleanup = terminal.get("cleanup")
    if (terminal.get("resource_status") == "COMPLETED"
            and type(terminal.get("OS_exit_code")) is int
            and terminal["OS_exit_code"] == 0
            and isinstance(cleanup, dict)
            and cleanup.get("status") == "NO_LIVE_GROUP_MEMBERS"
            and cleanup.get("live_remaining") == []):
        return _event(job, "COMPLETED", "TRANSPORT_AND_SAVED_OS_MATCH", rc, terminal)
    if terminal.get("resource_status") == "FAILED":
        return _event(job, "FAILED", "SAVED_OS_FAILED", rc, terminal)
    return _event(job, "UNRESOLVED", "SAVED_OS_NOT_PROVEN_COMPLETE", rc, terminal)


def observe_foreground(jobs, observer_seconds):
    """Yield one event per job as its handle/terminal read finishes, without polling.

    Callers must validate root authorization before Popen, use shell=False and
    durable logs, record the launch monotonic time, and bound read_terminal I/O.
    read_terminal is called once AFTER process.wait() returns, even on nonzero
    transport exit. Missing records are unresolved, never silently retried.

    The observer uses an absolute monotonic wall deadline. Already-queued events
    are handled first. A caller paused between yields can delay delivery; this
    generator cannot force the caller to resume iteration by the wall deadline.
    A blocked reader runs in a daemon thread; it is not killed.
    No wait handle is discarded in the returned event: pid/OS location identify
    unresolved work for an explicit follow-up, not an automatic relaunch.
    """
    _positive(observer_seconds, "observer_seconds")
    jobs = list(jobs)
    if len({j.name for j in jobs}) != len(jobs):
        raise ValueError("duplicate job name")
    if len({j.process.pid for j in jobs}) != len(jobs):
        raise ValueError("duplicate process handle")
    for job in jobs:
        job.validate()
    deadline = time.monotonic() + observer_seconds
    completed = queue.Queue()

    def worker(job):
        try:
            # wait() handles both already-exited and newly-exiting children.
            if job.process.poll() is not None:
                rc = job.process.returncode
            else:
                remaining = min(deadline, job.started_monotonic + job.transport_seconds) - time.monotonic()
                if remaining <= 0:
                    completed.put(_event(job, "UNRESOLVED", "TRANSPORT_DEADLINE"))
                    return
                rc = job.process.wait(timeout=remaining)
            terminal = job.read_terminal()
            event = _assess(job, rc, terminal)
        except subprocess.TimeoutExpired:
            event = _event(job, "UNRESOLVED", "TRANSPORT_DEADLINE")
        except Exception as exc:
            event = _event(job, "UNRESOLVED", "OBSERVER_OR_TERMINAL_READ_ERROR: " + repr(exc),
                           job.process.returncode)
        if time.monotonic() > deadline:
            event = _event(job, "UNRESOLVED", "OBSERVER_DEADLINE", job.process.returncode)
        completed.put(event)

    pending = {j.name: j for j in jobs}
    for job in jobs:
        threading.Thread(target=worker, args=(job,), name="completion:" + job.name,
                         daemon=True).start()
    while pending:
        remaining = deadline - time.monotonic()
        try:
            # A completion queued before registration or at the deadline wins.
            event = completed.get_nowait()
        except queue.Empty:
            if remaining <= 0:
                for job in pending.values():
                    yield _event(job, "UNRESOLVED", "OBSERVER_DEADLINE", job.process.returncode)
                return
            try:
                event = completed.get(timeout=remaining)
            except queue.Empty:
                continue
        if event["job"] in pending:
            del pending[event["job"]]
            yield event
