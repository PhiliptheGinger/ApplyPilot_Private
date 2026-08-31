"""Conservative safety-net watchdog for ApplyPilot runs on this machine.

2026-08-30: extended after a real OOM crash (exit code -536870904) during
`run score tailor cover --stream --limit 20`, and a repeat OOM when
terminating a subsequent `--limit 1` run. The machine has only ~11.8 GB
total RAM, an HDD that has shown sustained 100% active time at very low
throughput, and Ollama installed (idle footprint ~20-40 MB measured, but
has previously loaded models using several GB). The original version of
this script only ever watched disk latency and only ever launched
`run discover` -- it had no RAM awareness at all, which is the metric that
actually correlates with the OOM crashes being investigated. Changes made
here are additive: RAM monitoring, diagnostic process logging when
pressure builds, an optional Ollama stop, and a configurable target
command (via argv) so this can supervise whatever `applypilot run ...`
invocation is actually in use today, not just discovery.

Deliberately NOT changed: the disk-pressure detection logic, the
terminate-then-kill process tree shutdown, and the post-emergency sleep --
all already worked and are out of scope for this pass.
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

import psutil


PROJECT_DIR = Path(r"C:\Users\phili\Projects\resume-agent")
LOG_DIR = PROJECT_DIR / "watchdog_logs"

SAMPLE_SECONDS = 2
TRIGGER_SAMPLES = 10  # 10 * 2s = 20s of sustained severe pressure before acting
DISK_LATENCY_THRESHOLD_MS = 1000.0
DISK_ACTIVE_TIME_THRESHOLD = 99.0

# 2026-08-30: real observation that the PowerShell/WMI disk-stat subprocess
# call (get_disk_stats) occasionally took >10s during heavy I/O -- and was
# being called on every single 2s tick, sharing one try/except with the RAM
# check. Two real bugs from that: (1) a slow-but-successful disk call
# stretched the ENTIRE sampling interval well past SAMPLE_SECONDS, so the
# "TRIGGER_SAMPLES * SAMPLE_SECONDS = 20s sustained pressure" contract could
# silently balloon to 60-120s+ under exactly the disk-thrashing conditions
# that matter most; (2) when the call actually exceeded its own 10s timeout
# (subprocess.TimeoutExpired), that exception propagated out of
# get_disk_stats() into the SAME try block computing RAM badness -- so the
# RAM reading fetched moments earlier was discarded and never counted
# toward bad_samples for that sample, i.e. a disk-stat failure could
# silently suppress real, concurrent RAM-pressure detection. Fix (see
# main()): RAM is now sampled and counted every tick, fully independent of
# disk-stat outcome; disk-stat itself is now throttled to run only once per
# DISK_STAT_SAMPLE_EVERY ticks so a slow/stuck PowerShell call can only
# delay a fraction of samples, and repeated PowerShell process creation
# (itself disk/CPU churn on an already-strained HDD) drops 5x.
DISK_STAT_SAMPLE_EVERY = 5

# 2026-08-30: raised from an initial 750 MB after a real incident review.
# `run score tailor cover --stream --limit 20` OOM'd almost immediately;
# `--limit 1` survived for a long stretch and only OOM'd when the run was
# being TERMINATED. Direct measurement on this machine (baseline, no
# ApplyPilot running) found only ~3.6 GB free with ~8.2 GB already used by
# Windows/VS Code/Edge/Claude/Memory Compression -- i.e. the headroom above
# 750 MB was itself only ~2.8 GB even at rest, and streaming mode runs
# THREE stage threads (score/tailor/cover) concurrently, each independently
# holding a job description, prompt, and LLM response in memory at once.
# 750 MB requiring a full 20s sustained dwell (10 samples @ 2s) was cutting
# it close on a machine that can plausibly drop several hundred MB/s once
# genuinely under pressure -- there may not have been 20 full seconds
# between "clearly in trouble" and "already gone". 1400 MB (~12% of total)
# is comfortably above this machine's normal at-rest free-RAM range (~3.6 GB
# measured), so it will not fire during ordinary background load, while
# giving meaningfully more lead time before an actual allocation failure.
RAM_AVAILABLE_THRESHOLD_MB = 2000.0

# Idle Ollama (server + tray app, no model loaded) measured on this machine
# at ~20-40 MB combined. A loaded model is gigabytes. This threshold is what
# distinguishes "Ollama happens to be running" (leave it alone -- it is not
# part of the problem) from "Ollama has a model loaded and is a plausible
# contributor" (stop it too) -- see the "optionally stop Ollama" requirement.
OLLAMA_RAM_THRESHOLD_MB = 500.0

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"watchdog_{datetime.now():%Y%m%d_%H%M%S}.log"


def log(message):
	line = f"{datetime.now():%Y-%m-%d %H:%M:%S.%f} {message}"
	print(line, flush=True)
	with LOG_FILE.open("a", encoding="utf-8") as file:
		file.write(line + "\n")


def get_disk_stats():
	result = subprocess.run(
		[
			"powershell.exe",
			"-NoProfile",
			"-Command",
			(
				"Get-CimInstance "
				"Win32_PerfFormattedData_PerfDisk_PhysicalDisk "
				"| Where-Object { $_.Name -eq '0 C:' } "
				"| Select-Object PercentDiskTime, "
				"AvgDiskSecPerTransfer, "
				"DiskReadBytesPerSec, "
				"DiskWriteBytesPerSec "
				"| ConvertTo-Json -Compress"
			),
		],
		capture_output=True,
		text=True,
		timeout=5,
	)

	if result.returncode != 0 or not result.stdout.strip():
		return None

	data = json.loads(result.stdout)

	return {
		"active": float(data["PercentDiskTime"]),
		"latency_ms": float(data["AvgDiskSecPerTransfer"]) * 1000,
		"read_mb": float(data["DiskReadBytesPerSec"]) / 1024 / 1024,
		"write_mb": float(data["DiskWriteBytesPerSec"]) / 1024 / 1024,
	}


def get_ram_stats():
	vm = psutil.virtual_memory()
	return {
		"available_mb": vm.available / 1024 / 1024,
		"percent_used": vm.percent,
	}


def log_process_diagnostics(top_n=8):
	"""Snapshot of the top RAM-consuming processes -- written to the log
	whenever pressure is building, so a post-mortem has something concrete
	to look at besides the aggregate disk/RAM numbers."""
	procs = []
	for proc in psutil.process_iter(["pid", "name", "memory_info"]):
		try:
			mem_mb = proc.info["memory_info"].rss / 1024 / 1024
			procs.append((mem_mb, proc.info["pid"], proc.info["name"]))
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			continue
	procs.sort(reverse=True)
	log("Top processes by RAM:")
	for mem_mb, pid, name in procs[:top_n]:
		log(f"    {mem_mb:8.1f} MB  PID {pid:6d}  {name}")


def find_ollama_processes():
	found = []
	for proc in psutil.process_iter(["pid", "name", "memory_info"]):
		try:
			name = (proc.info["name"] or "").lower()
			if "ollama" in name:
				mem_mb = proc.info["memory_info"].rss / 1024 / 1024
				found.append((proc, mem_mb))
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			continue
	return found


def stop_ollama_if_contributing():
	"""Optionally stop Ollama -- only when it is clearly using substantial
	RAM (a loaded model), never just because it happens to be running."""
	ollama_procs = find_ollama_processes()
	if not ollama_procs:
		log("Ollama not running -- nothing to stop.")
		return

	total_mb = sum(mem for _, mem in ollama_procs)
	if total_mb < OLLAMA_RAM_THRESHOLD_MB:
		log(
			f"Ollama running but only using {total_mb:.1f} MB "
			f"(< {OLLAMA_RAM_THRESHOLD_MB:.0f} MB threshold) -- leaving it alone, "
			"not implicated in this emergency."
		)
		return

	log(f"Ollama using {total_mb:.1f} MB -- stopping it as a contributing factor.")
	for proc, mem_mb in ollama_procs:
		try:
			log(f"Terminating Ollama PID {proc.pid} ({mem_mb:.1f} MB)")
			proc.terminate()
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			pass

	_, alive = psutil.wait_procs([p for p, _ in ollama_procs], timeout=5)

	for proc in alive:
		try:
			log(f"Force-killing Ollama PID {proc.pid}")
			proc.kill()
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			pass


DEFAULT_APPLYPILOT_ARGS = ["run", "discover", "--workers", "1", "--quiet"]


def resolve_applypilot_args(argv):
	"""Which `applypilot` command this watchdog run should supervise.

	`argv` is everything after the script name (i.e. sys.argv[1:]) --
	forwarded verbatim to `applypilot` when present, so this can supervise
	whatever invocation is actually in use (e.g. `run score tailor cover
	--stream --limit 20`) without editing the script. Falls back to the
	original hardcoded default when no arguments are given, so existing
	invocations keep working unchanged."""
	return list(argv) if argv else list(DEFAULT_APPLYPILOT_ARGS)


def kill_applypilot_tree(pid):
	try:
		process = psutil.Process(pid)
	except psutil.NoSuchProcess:
		log("ApplyPilot process already exited.")
		return

	children = process.children(recursive=True)

	for child in children:
		try:
			log(f"Terminating child PID {child.pid}: {child.name()}")
			child.terminate()
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			pass

	try:
		log(f"Terminating ApplyPilot PID {pid}: {process.name()}")
		process.terminate()
	except (psutil.NoSuchProcess, psutil.AccessDenied):
		pass

	_, alive = psutil.wait_procs(children + [process], timeout=5)

	for child in alive:
		try:
			log(f"Force-killing PID {child.pid}: {child.name()}")
			child.kill()
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			pass


def main():
	args = resolve_applypilot_args(sys.argv[1:])

	log("=== ApplyPilot watchdog starting ===")
	log(f"Project: {PROJECT_DIR}")
	log(f"Command: applypilot {' '.join(args)}")
	log(f"Log file: {LOG_FILE}")
	log(
		f"Thresholds: disk active>={DISK_ACTIVE_TIME_THRESHOLD:.0f}% AND "
		f"latency>={DISK_LATENCY_THRESHOLD_MS:.0f}ms, OR RAM available<"
		f"{RAM_AVAILABLE_THRESHOLD_MB:.0f}MB -- sustained {TRIGGER_SAMPLES} "
		f"consecutive samples ({TRIGGER_SAMPLES * SAMPLE_SECONDS}s) before acting."
	)

	os.chdir(PROJECT_DIR)

	applypilot = subprocess.Popen(
		[sys.executable, "-m", "applypilot", *args],
		cwd=PROJECT_DIR,
	)

	log(f"ApplyPilot PID: {applypilot.pid}")

	bad_samples = 0
	sample_count = 0
	last_disk_bad = False

	while applypilot.poll() is None:
		time.sleep(SAMPLE_SECONDS)
		sample_count += 1

		try:
			# RAM is sampled and evaluated every tick, fully independent of
			# disk-stat below -- a slow/failed/timed-out disk query must
			# never suppress RAM-pressure detection (see DISK_STAT_SAMPLE_EVERY
			# comment above for the incident this fixes).
			ram = get_ram_stats()
			log(
				f"RAM available={ram['available_mb']:.0f} MB "
				f"used={ram['percent_used']:.1f}%"
			)
			ram_bad = ram["available_mb"] < RAM_AVAILABLE_THRESHOLD_MB

			# Disk-stat spawns a PowerShell/WMI subprocess that has been
			# observed to occasionally take >10s under real I/O pressure --
			# throttled so it can only delay a fraction of samples and so it
			# doesn't itself add unnecessary repeated process creation on an
			# already-strained HDD. On skipped ticks, reuse the last known
			# disk_bad reading rather than treating it as "unknown" so a
			# genuine sustained disk emergency is still detected between
			# checks.
			disk_bad = last_disk_bad
			if sample_count % DISK_STAT_SAMPLE_EVERY == 1:
				try:
					disk = get_disk_stats()
				except Exception as exc:
					log(f"WARNING: disk-stat query failed/timed out: {exc}")
					disk = None

				if disk is None:
					log("WARNING: Could not obtain Disk 0 performance data.")
					disk_bad = False
				else:
					log(
						f"Disk0 active={disk['active']:.1f}% "
						f"read={disk['read_mb']:.2f} MB/s "
						f"write={disk['write_mb']:.2f} MB/s "
						f"latency={disk['latency_ms']:.1f} ms"
					)
					disk_bad = (
						disk["active"] >= DISK_ACTIVE_TIME_THRESHOLD
						and disk["latency_ms"] >= DISK_LATENCY_THRESHOLD_MS
					)
				last_disk_bad = disk_bad

			# Combined counter: either cause counts toward the same sustained
			# window. The goal is "has the machine been under severe
			# resource pressure continuously for TRIGGER_SAMPLES samples",
			# not "has the SAME metric specifically been bad the whole
			# time" -- a machine flipping between disk-bound and RAM-bound
			# distress for 20s straight is just as much an emergency as one
			# stuck on a single metric.
			if disk_bad or ram_bad:
				bad_samples += 1
				reasons = []
				if disk_bad:
					reasons.append("disk")
				if ram_bad:
					reasons.append("RAM")
				log(
					f"WARNING: severe pressure ({'+'.join(reasons)}) "
					f"({bad_samples}/{TRIGGER_SAMPLES})"
				)
				log_process_diagnostics()
			else:
				bad_samples = 0

			if bad_samples >= TRIGGER_SAMPLES:
				log("!!! EMERGENCY CONDITION TRIGGERED !!!")
				log(
					f"Sustained severe pressure for "
					f"{TRIGGER_SAMPLES * SAMPLE_SECONDS}s "
					f"(disk_bad={disk_bad}, ram_bad={ram_bad})."
				)
				log("Final process snapshot before shutdown:")
				log_process_diagnostics(top_n=15)

				log("Stopping ApplyPilot process tree...")
				kill_applypilot_tree(applypilot.pid)

				stop_ollama_if_contributing()

				log("Attempting to put Windows to sleep...")
				try:
					# 2026-08-30 fix: real incident -- `SetSuspendState` did
					# not return on this machine (confirmed live: the
					# rundll32.exe child and this watchdog process were both
					# still alive and blocked here 3+ minutes after this line
					# logged, with ApplyPilot already successfully killed).
					# subprocess.run's own timeout handling kills the child
					# process before re-raising TimeoutExpired, so this can't
					# leave an orphaned rundll32.exe behind either. The
					# emergency's actually load-bearing step (killing
					# ApplyPilot, above) has already completed by this point
					# regardless of whether sleep succeeds -- a bounded
					# timeout here can only ever improve on "hangs forever",
					# never regress a case that previously worked.
					subprocess.run(
						[
							"rundll32.exe",
							"powrprof.dll,SetSuspendState",
							"0,1,0",
						],
						check=False,
						timeout=15,
					)
					log("Sleep command issued.")
				except subprocess.TimeoutExpired:
					log("WARNING: sleep command did not return within 15s -- abandoning it (ApplyPilot is already stopped).")
				except Exception as exc:
					log(f"ERROR entering sleep: {exc}")

				return 0

		except Exception as exc:
			log(f"Monitoring error: {exc}")

	log(f"ApplyPilot exited normally with code {applypilot.returncode}")
	log("=== Watchdog finished ===")
	return applypilot.returncode


if __name__ == "__main__":
	sys.exit(main())
