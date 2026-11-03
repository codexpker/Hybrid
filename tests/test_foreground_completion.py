"""Synthetic local subprocess fixtures only; no real SSH/science/data access."""
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

from hybrid_tsad.foreground_completion import ForegroundJob, observe_foreground, _ssh_keepalive


class ForegroundCompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.processes = []
        self.identity = {"release_sha256": "a" * 64, "config_sha256": "b" * 64}

    def tearDown(self):
        # Test-owned local fixtures only. The observer itself never kills.
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=3)
            if p.stdout is not None:
                p.stdout.close()
        self.tmp.cleanup()

    def spawn(self, name, delay=0, rc=0, save=True, failed=False, pipe=False, timeout=3):
        path = self.root / (name + ".OS_WAIT.json")
        record = dict(self.identity, OS_exit_code=1 if failed else 0,
                      resource_status="FAILED" if failed else "COMPLETED",
                      cleanup={"status": "NO_LIVE_GROUP_MEMBERS", "live_remaining": []})
        code = ("import pathlib,time,json,os;time.sleep(" + repr(delay) + ");"
                + ("pathlib.Path(" + repr(str(path)) + ").write_text(" + repr(json.dumps(record)) + ");" if save else "")
                + "os._exit(" + repr(rc) + ")")
        argv = [sys.executable, "-B", "-c", code]
        start = time.monotonic()
        p = subprocess.Popen(argv, stdout=subprocess.PIPE if pipe else subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, shell=False)
        self.processes.append(p)
        def read():
            return json.loads(path.read_text()) if path.exists() else None
        return ForegroundJob(name, p, tuple(argv), "SYNTHETIC_ROOT_FIXTURE_ONLY",
                             (str(path),), read, dict(self.identity), timeout, start)

    def test_three_concurrent_fast_event_precedes_slowest(self):
        jobs = [self.spawn("slow", .7), self.spawn("fast", .03), self.spawn("middle", .3)]
        events = observe_foreground(jobs, 3)
        first = next(events)
        self.assertEqual(first["job"], "fast")
        self.assertEqual(first["status"], "COMPLETED")
        self.assertIsNone(jobs[0].process.poll(), "fast event must not wait for slowest")
        self.assertEqual({e["job"] for e in [first] + list(events)}, {j.name for j in jobs})

    def test_already_exited_before_registration(self):
        job = self.spawn("already")
        job.process.wait(timeout=3)
        events = list(observe_foreground([job], 1))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "COMPLETED")

    def test_nonzero_transport_with_saved_success_is_unresolved(self):
        e = list(observe_foreground([self.spawn("nonzero", rc=7)], 2))[0]
        self.assertEqual(e["status"], "UNRESOLVED")
        self.assertEqual(e["transport_returncode"], 7)
        self.assertIn("TRANSPORT_NONZERO", e["reason"])

    def test_disconnect_like_exit_without_terminal_is_unresolved(self):
        e = list(observe_foreground([self.spawn("disconnect", rc=255, save=False)], 2))[0]
        self.assertEqual(e["status"], "UNRESOLVED")
        self.assertEqual(e["transport_returncode"], 255)
        self.assertFalse(e["automatic_retry"])

    def test_transport_deadline_does_not_kill_or_retry(self):
        job = self.spawn("timeout", delay=.8, timeout=.04)
        e = list(observe_foreground([job], 2))[0]
        self.assertEqual(e["reason"], "TRANSPORT_DEADLINE")
        self.assertEqual(e["status"], "UNRESOLVED")
        self.assertIsNone(job.process.poll())
        self.assertEqual(len(self.processes), 1)
        self.assertFalse(e["process_terminated_by_observer"])
        self.assertEqual(e["terminal_locations"], list(job.terminal_locations))

    def test_observer_deadline_bounds_blocked_terminal_reader(self):
        from dataclasses import replace
        job = self.spawn("blockedreader")
        job.process.wait(timeout=3)
        def blocked():
            time.sleep(.5)
            return None
        job = replace(job, read_terminal=blocked)
        start = time.monotonic()
        e = list(observe_foreground([job], .05))[0]
        self.assertLess(time.monotonic() - start, .4)
        self.assertEqual(e["reason"], "OBSERVER_DEADLINE")
        self.assertEqual(e["status"], "UNRESOLVED")

    def test_zero_transport_without_saved_terminal_not_success(self):
        e = list(observe_foreground([self.spawn("missing", save=False)], 2))[0]
        self.assertEqual(e["reason"], "TERMINAL_RECORD_ABSENT")
        self.assertEqual(e["status"], "UNRESOLVED")

    def test_scientific_failure_despite_supervisor_exit_zero(self):
        e = list(observe_foreground([self.spawn("scientificfailed", failed=True)], 2))[0]
        self.assertEqual(e["transport_returncode"], 0)
        self.assertEqual(e["status"], "FAILED")

    def test_terminal_root_identity_mismatch_never_success(self):
        from dataclasses import replace
        job = self.spawn("identity")
        job = replace(job, expected_terminal_identity={"release_sha256": "c" * 64})
        e = list(observe_foreground([job], 2))[0]
        self.assertEqual(e["status"], "UNRESOLVED")
        self.assertEqual(e["reason"], "TERMINAL_IDENTITY_MISMATCH")

    def test_contradictory_cleanup_with_live_process_never_success(self):
        job = self.spawn("contradictory_cleanup")
        job.process.wait(timeout=3)
        path = pathlib.Path(job.terminal_locations[0])
        record = json.loads(path.read_text())
        record["cleanup"]["live_remaining"] = [{"pid": 999999, "synthetic": True}]
        path.write_text(json.dumps(record))
        e = list(observe_foreground([job], 2))[0]
        self.assertEqual(e["transport_returncode"], 0)
        self.assertEqual(e["status"], "UNRESOLVED")
        self.assertEqual(e["reason"], "SAVED_OS_NOT_PROVEN_COMPLETE")

    def test_pipe_rejected(self):
        job = self.spawn("pipe", pipe=True)
        with self.assertRaisesRegex(ValueError, "PIPE"):
            list(observe_foreground([job], 2))

    def test_unapproved_argv_rejected(self):
        from dataclasses import replace
        job = self.spawn("argv")
        job = replace(job, approved_argv=("unapproved",))
        with self.assertRaisesRegex(ValueError, "differs"):
            list(observe_foreground([job], 2))

    def test_ssh_keepalive_requires_positive_bounded_options_without_network(self):
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                "-T", "synthetic.invalid", "approved-remote-command"]
        _ssh_keepalive(argv)
        with self.assertRaisesRegex(ValueError, "serveralivecountmax"):
            _ssh_keepalive(argv[:7] + argv[9:])
        bad = list(argv)
        bad[8] = "ServerAliveCountMax=0"
        with self.assertRaisesRegex(ValueError, "serveralivecountmax"):
            _ssh_keepalive(bad)


if __name__ == "__main__":
    unittest.main()
