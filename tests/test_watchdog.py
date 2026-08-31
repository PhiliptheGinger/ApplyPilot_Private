"""Focused tests for the 2026-08-30 watchdog changes (repo-root watchdog.py,
not part of the applypilot package).

Context: this machine has ~11.8 GB total RAM and has produced a real OOM
crash (exit code -536870904) running ApplyPilot streaming stages. The
watchdog previously only monitored disk latency and only ever launched
`run discover` -- it had no RAM awareness (the metric that actually
correlates with the observed crashes) and could not supervise a different
command without editing the script. These tests cover the new/changed
logic only: RAM-threshold decision-making, the Ollama-stop threshold
decision (mocked -- must never actually terminate a real process in a
test), and the configurable-command resolution. Disk-stats/process-tree
logic is unchanged from the pre-existing script and is not re-tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import watchdog


class TestResolveApplypilotArgs:
	def test_no_argv_falls_back_to_default(self):
		assert watchdog.resolve_applypilot_args([]) == watchdog.DEFAULT_APPLYPILOT_ARGS

	def test_argv_forwarded_verbatim(self):
		argv = ["run", "score", "tailor", "cover", "--stream", "--limit", "20"]
		assert watchdog.resolve_applypilot_args(argv) == argv

	def test_returns_a_new_list_not_a_reference_to_the_default(self):
		"""Mutating the result must never corrupt DEFAULT_APPLYPILOT_ARGS for
		a later call with no argv."""
		result = watchdog.resolve_applypilot_args([])
		result.append("--mutated")
		assert watchdog.resolve_applypilot_args([]) == watchdog.DEFAULT_APPLYPILOT_ARGS


class TestRamStats:
	def test_get_ram_stats_returns_expected_shape(self):
		"""Real (read-only, safe) call -- no mocking needed."""
		stats = watchdog.get_ram_stats()
		assert "available_mb" in stats
		assert "percent_used" in stats
		assert stats["available_mb"] > 0
		assert 0 <= stats["percent_used"] <= 100

	def test_ram_threshold_is_a_small_but_nonzero_fraction_of_a_low_ram_machine(self):
		"""Sanity guard on the constant itself -- this machine has ~11.8 GB
		total RAM; the threshold must be low enough to avoid constant false
		triggers but high enough to act before real exhaustion."""
		assert 0 < watchdog.RAM_AVAILABLE_THRESHOLD_MB < 2000


class TestStopOllamaIfContributing:
	"""All process-affecting calls are mocked -- this must never terminate a
	real process during a test run."""

	def _fake_proc(self, pid=999):
		proc = MagicMock()
		proc.pid = pid
		return proc

	def test_ollama_not_running_is_a_noop(self, monkeypatch):
		monkeypatch.setattr(watchdog, "find_ollama_processes", list)
		watchdog.stop_ollama_if_contributing()  # must not raise

	def test_idle_ollama_under_threshold_is_left_alone(self, monkeypatch):
		proc = self._fake_proc()
		monkeypatch.setattr(
			watchdog, "find_ollama_processes", lambda: [(proc, watchdog.OLLAMA_RAM_THRESHOLD_MB - 100)]
		)
		watchdog.stop_ollama_if_contributing()
		proc.terminate.assert_not_called()
		proc.kill.assert_not_called()

	def test_loaded_ollama_over_threshold_is_terminated(self, monkeypatch):
		proc = self._fake_proc()
		monkeypatch.setattr(
			watchdog, "find_ollama_processes", lambda: [(proc, watchdog.OLLAMA_RAM_THRESHOLD_MB + 2000)]
		)
		# Simulate a clean exit after terminate() -- no force-kill needed.
		monkeypatch.setattr(watchdog.psutil, "wait_procs", lambda procs, timeout=None: ([], []))
		watchdog.stop_ollama_if_contributing()
		proc.terminate.assert_called_once()
		proc.kill.assert_not_called()

	def test_loaded_ollama_that_wont_die_gets_force_killed(self, monkeypatch):
		proc = self._fake_proc()
		monkeypatch.setattr(
			watchdog, "find_ollama_processes", lambda: [(proc, watchdog.OLLAMA_RAM_THRESHOLD_MB + 2000)]
		)
		# Simulate the process still being alive after the graceful wait.
		monkeypatch.setattr(watchdog.psutil, "wait_procs", lambda procs, timeout=None: ([], [proc]))
		watchdog.stop_ollama_if_contributing()
		proc.terminate.assert_called_once()
		proc.kill.assert_called_once()


class TestLogProcessDiagnostics:
	def test_does_not_raise(self):
		"""Real (read-only) call over the actual process list -- just needs
		to complete without error and not crash the monitoring loop."""
		watchdog.log_process_diagnostics(top_n=3)


class _FakePopen:
	"""Stands in for the real `applypilot` child process. poll() always
	returns None ("still running") -- main()'s loop is expected to exit via
	the emergency-shutdown `return 0` branch, not via child exit, in these
	tests."""

	def __init__(self, *args, **kwargs):
		self.pid = 999999

	def poll(self):
		return None


class TestMainLoopDiskRamDecoupling:
	"""2026-08-30 fix: a real observation that the disk-stat PowerShell/WMI
	subprocess call occasionally took >10s under heavy I/O, and was being
	called on every 2s tick sharing one try/except with the RAM check --
	meaning a disk-stat failure/timeout silently discarded that same
	sample's already-fetched RAM reading and never counted it toward
	bad_samples. These tests drive watchdog.main() end-to-end (with all
	process-affecting calls mocked -- must never touch a real process or
	actually suspend the machine) to prove: (1) sustained RAM-only pressure
	still triggers the emergency shutdown even when disk-stat fails on
	EVERY call, and (2) disk-stat is now throttled (DISK_STAT_SAMPLE_EVERY)
	rather than invoked on every single sample."""

	def _install_common_mocks(self, monkeypatch, get_disk_stats):
		monkeypatch.setattr(watchdog.subprocess, "Popen", _FakePopen)
		monkeypatch.setattr(watchdog.time, "sleep", lambda *_a, **_kw: None)
		monkeypatch.setattr(
			watchdog,
			"get_ram_stats",
			lambda: {
				"available_mb": watchdog.RAM_AVAILABLE_THRESHOLD_MB - 100,
				"percent_used": 95.0,
			},
		)
		monkeypatch.setattr(watchdog, "get_disk_stats", get_disk_stats)
		kill_mock = MagicMock()
		ollama_mock = MagicMock()
		run_mock = MagicMock()
		monkeypatch.setattr(watchdog, "kill_applypilot_tree", kill_mock)
		monkeypatch.setattr(watchdog, "stop_ollama_if_contributing", ollama_mock)
		monkeypatch.setattr(watchdog.subprocess, "run", run_mock)
		monkeypatch.setattr(sys, "argv", ["watchdog.py"])
		return kill_mock, ollama_mock, run_mock

	def test_sustained_ram_pressure_triggers_even_when_disk_stat_always_fails(self, monkeypatch):
		"""The exact real-incident shape: disk-stat times out every time.
		Before the fix, this exception was caught by the SAME try block
		computing ram_bad, so a disk-stat failure discarded that sample's
		RAM reading entirely -- sustained RAM pressure could never
		accumulate bad_samples to TRIGGER_SAMPLES. After the fix, RAM is
		evaluated independently of disk-stat's outcome."""

		def _always_times_out():
			raise watchdog.subprocess.TimeoutExpired(cmd="powershell.exe", timeout=5)

		kill_mock, ollama_mock, run_mock = self._install_common_mocks(monkeypatch, _always_times_out)

		result = watchdog.main()

		assert result == 0
		kill_mock.assert_called_once()
		ollama_mock.assert_called_once()
		run_mock.assert_called_once()  # the emergency sleep command

	def test_emergency_sleep_timeout_does_not_hang_main(self, monkeypatch):
		"""Real incident: `SetSuspendState` did not return on this machine --
		confirmed live, the watchdog process and its rundll32.exe child were
		both still alive 3+ minutes after "Attempting to put Windows to
		sleep..." was logged, with ApplyPilot already successfully killed.
		The sleep subprocess.run call had no timeout at all. main() must
		still return promptly (not hang, not raise) when the sleep call
		times out -- killing ApplyPilot is the load-bearing action and has
		already completed by that point regardless of whether sleep works."""

		def _always_times_out_disk_stat():
			raise watchdog.subprocess.TimeoutExpired(cmd="powershell.exe", timeout=5)

		kill_mock, ollama_mock, _run_mock = self._install_common_mocks(monkeypatch, _always_times_out_disk_stat)

		def _sleep_call_hangs(*args, **kwargs):
			if kwargs.get("timeout") is None:
				pytest.fail("emergency sleep subprocess.run call must pass an explicit timeout")
			raise watchdog.subprocess.TimeoutExpired(cmd=args[0] if args else "rundll32.exe", timeout=kwargs["timeout"])

		monkeypatch.setattr(watchdog.subprocess, "run", _sleep_call_hangs)

		result = watchdog.main()

		assert result == 0
		kill_mock.assert_called_once()
		ollama_mock.assert_called_once()

	def test_disk_stat_is_throttled_not_called_every_sample(self, monkeypatch):
		"""Over the TRIGGER_SAMPLES=10 samples needed to trigger (at 2s/sample
		nominal cadence), disk-stat must only be queried on the throttled
		schedule (DISK_STAT_SAMPLE_EVERY), not on all 10 -- both to bound how
		often a slow/stuck PowerShell call can stretch the loop and to cut
		down repeated PowerShell process creation on an already-strained HDD."""
		call_count = 0

		def _counting_disk_stats():
			# "could not obtain data" -- not disk_bad, doesn't affect trigger path
			nonlocal call_count
			call_count += 1

		self._install_common_mocks(monkeypatch, _counting_disk_stats)

		watchdog.main()

		# TRIGGER_SAMPLES=10 RAM-bad samples fire the emergency exactly at
		# sample_count == TRIGGER_SAMPLES; disk-stat only runs when
		# sample_count % DISK_STAT_SAMPLE_EVERY == 1, i.e. samples 1 and 6
		# for the current DISK_STAT_SAMPLE_EVERY=5 / TRIGGER_SAMPLES=10.
		expected_calls = len(
			[n for n in range(1, watchdog.TRIGGER_SAMPLES + 1) if n % watchdog.DISK_STAT_SAMPLE_EVERY == 1]
		)
		assert call_count == expected_calls
		assert call_count < watchdog.TRIGGER_SAMPLES
