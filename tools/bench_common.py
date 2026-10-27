"""Measure a fresh Rust service/worker, keeping the Windows file cache intact.

No recorded audio or recognized text is printed. Detailed reports stay local in
evidence/. A Job Object owns the benchmark tree and kills it on timeout/cleanup.
This measures process-cold/model-cold starts, not reboot-cold disk performance.
"""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
import statistics
import struct
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("duration", "text", "text_accu", "tokens", "timestamps")
MEMORY_FIELDS = ("private_bytes", "private_working_set", "working_set")
CREATE_NO_WINDOW = 0x08000000
KILL_ON_JOB_CLOSE = 0x2000
k = ctypes.WinDLL("kernel32", use_last_error=True)
p = ctypes.WinDLL("psapi", use_last_error=True)


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", w.DWORD), ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t), ("active_limit", w.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", w.DWORD),
                ("scheduling", w.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("reads", "writes", "others", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class ProcessIds(ctypes.Structure):
    _fields_ = [("assigned", w.DWORD), ("count", w.DWORD),
                ("ids", ctypes.c_size_t * 128)]


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("page_faults", w.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
            "pagefile", "peak_pagefile", "private", "private_ws", "shared_commit")]


k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
k.CreateJobObjectW.restype = w.HANDLE
k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
k.SetInformationJobObject.restype = w.BOOL
k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
k.QueryInformationJobObject.restype = w.BOOL
k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
k.AssignProcessToJobObject.restype = w.BOOL
k.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
k.TerminateJobObject.restype = w.BOOL
k.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]
k.IsProcessInJob.restype = w.BOOL
k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
k.OpenProcess.restype = w.HANDLE
k.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(ctypes.c_uint64)] * 4
k.GetProcessTimes.restype = w.BOOL
k.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
k.GetExitCodeProcess.restype = w.BOOL
k.CloseHandle.argtypes = [w.HANDLE]
k.CloseHandle.restype = w.BOOL
p.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(MemoryCounters), w.DWORD]
p.GetProcessMemoryInfo.restype = w.BOOL


class OwnedTree:
    def __init__(self):
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, proc):
        # serve does not start its worker until the first request is sent. Assign
        # the root before sending any data, so every worker inherits this job.
        if not k.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def ids(self):
        ids = ProcessIds()
        if not k.QueryInformationJobObject(self.handle, 3, ctypes.byref(ids), ctypes.sizeof(ids), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if ids.count > len(ids.ids) or ids.assigned > len(ids.ids):
            raise RuntimeError("Unexpectedly large benchmark process tree")
        return list(ids.ids[:ids.count])

    def _snapshot_target_gone(self, pid, handle=None):
        if handle:
            exit_code = w.DWORD()
            if k.GetExitCodeProcess(handle, ctypes.byref(exit_code)) and exit_code.value != 259:
                return True
        return pid not in self.ids()

    def _snapshot_failure(self, pid, operation, code, handle=None):
        if self._snapshot_target_gone(pid, handle):
            return
        raise OSError(code, f"{operation} failed for live owned PID {pid} (Win32 {code})")

    def snapshot(self):
        rows, exited = [], []
        expected = self.ids()
        for pid in expected:
            handle = k.OpenProcess(0x1000, False, pid)
            if not handle:
                self._snapshot_failure(pid, "OpenProcess", ctypes.get_last_error())
                exited.append(pid)
                continue
            try:
                member = w.BOOL()
                if not k.IsProcessInJob(handle, self.handle, ctypes.byref(member)):
                    self._snapshot_failure(pid, "IsProcessInJob", ctypes.get_last_error(), handle)
                    exited.append(pid)
                    continue
                if not member.value:
                    self._snapshot_failure(pid, "IsProcessInJob membership", 0, handle)
                    exited.append(pid)
                    continue
                info = MemoryCounters()
                info.cb = ctypes.sizeof(info)
                ticks = [ctypes.c_uint64() for _ in range(4)]
                if not p.GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
                    self._snapshot_failure(pid, "GetProcessMemoryInfo", ctypes.get_last_error(), handle)
                    exited.append(pid)
                    continue
                if not k.GetProcessTimes(handle, *[ctypes.byref(t) for t in ticks]):
                    self._snapshot_failure(pid, "GetProcessTimes", ctypes.get_last_error(), handle)
                    exited.append(pid)
                    continue
                rows.append({"pid": pid, "private_bytes": info.private,
                             "private_working_set": info.private_ws, "working_set": info.ws,
                             "peak_working_set": info.peak_ws,
                             "cpu_seconds": (ticks[2].value + ticks[3].value) / 1e7})
            finally:
                k.CloseHandle(handle)
        return {"processes": rows, "expected_process_ids": expected, "exited_process_ids": exited,
                "cpu_seconds_total": sum(row["cpu_seconds"] for row in rows),
                "totals": {key: sum(row[key] for row in rows) for key in MEMORY_FIELDS}}

    def kill(self):
        if self.handle and not k.TerminateJobObject(self.handle, 0xE0000002):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            k.CloseHandle(self.handle)
            self.handle = None


class Sampler:
    def __init__(self, owned, started, interval):
        self.owned, self.started, self.interval = owned, started, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="cold-memory-sampler", daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                value = self.owned.snapshot()
                value["elapsed_seconds"] = time.perf_counter() - self.started
                self.samples.append(value)
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set()
        self.thread.join()


def read_exact(stream, count):
    parts = []
    while count:
        data = stream.read(count)
        if not data:
            raise RuntimeError("Service pipe closed before a complete reply")
        parts.append(data)
        count -= len(data)
    return b"".join(parts)


def receive(proc):
    magic, size, audio = struct.unpack("<4sII", read_exact(proc.stdout, 12))
    if magic != b"YYUR" or audio or size > 64 * 1024 * 1024:
        raise RuntimeError("Invalid or oversized service reply")
    return json.loads(read_exact(proc.stdout, size))


def timed_io(owned, timeout, operation):
    expired = threading.Event()
    def terminate():
        expired.set()
        owned.kill()
    timer = threading.Timer(timeout, terminate)
    timer.daemon = True
    timer.start()
    try:
        return operation()
    finally:
        timer.cancel()
        timer.join()
        if expired.is_set():
            raise TimeoutError(f"Service operation exceeded {timeout:g}s; benchmark tree terminated")


def call(proc, owned, timeout, request, pcm=b""):
    def exchange():
        body = json.dumps(request, ensure_ascii=False).encode("utf-8")
        proc.stdin.write(struct.pack("<4sII", b"YYUR", len(body), len(pcm) // 4) + body + pcm)
        proc.stdin.flush()
        reply = receive(proc)
        if reply["id"] != request["id"] or reply.get("error"):
            raise RuntimeError("Service returned a failed or out-of-order reply; see local report/log")
        return reply
    return timed_io(owned, timeout, exchange)


def result_fields(reply):
    value = reply.get("result")
    if not isinstance(value, dict) or not value.get("text"):
        raise RuntimeError("Expected a nonempty recognition result")
    return {key: value[key] for key in FIELDS}


def result_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def run_round(args, item, pcm, index, expected):
    row = {"round": index + 1, "fixture_id": item["id"], "source": args.source,
           "compared_fields": list(FIELDS), "stages": [], "error": None, "passed": False}
    command = [str(args.exe), "--app-dir", str(args.app_dir), "--config", str(args.config),
               "--asr-idle-seconds", "0", "serve"]
    env = os.environ.copy()
    log_dir = ROOT / "evidence" / "logs" / args.tag / f"r{index + 1}"
    env["YYUR_LOG_DIR"] = str(log_dir)
    env.pop("YYUR_PROFILE_LOAD", None)
    if args.profile_load:
        env["YYUR_PROFILE_LOAD"] = "1"
    owned, proc, sampler = OwnedTree(), None, None
    stderr = ROOT / "evidence" / f"{args.tag}-r{index + 1}.stderr.txt"
    row["stderr_path"] = str(stderr)
    def stage(name):
        value = {"stage": name, "elapsed_seconds": time.perf_counter() - started,
                 **owned.snapshot()}
        row["stages"].append(value)
    try:
        with stderr.open("wb") as errors:
            started = time.perf_counter()
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=errors, env=env, creationflags=CREATE_NO_WINDOW)
            row["pid"] = proc.pid
            owned.assign(proc)
            sampler = Sampler(owned, started, args.sample_ms / 1000)
            sampler.thread.start()
            boot = timed_io(owned, args.timeout, lambda: receive(proc))
            row["boot_seconds"] = time.perf_counter() - started
            if boot.get("kind") != "boot" or boot.get("id") != 0:
                raise RuntimeError("Invalid service boot handshake")
            stage("boot")
            if not args.no_warmup:
                tick = time.perf_counter()
                ready = call(proc, owned, args.timeout, {"command": "warmup", "id": 1})
                row["warmup_seconds"] = time.perf_counter() - tick
                row["model_ready_seconds"] = time.perf_counter() - started
                row["warmup_metrics"] = ready.get("metrics")
                stage("model_ready")
            for request_id, name in ((2, "first"), (3, "warm")):
                request = {"command": "segment", "id": request_id,
                           "task_id": f"cold-r{index + 1}-{name}", "source": args.source,
                           "language": "auto", "context": "", "offset": 0.0, "overlap": 4.0,
                           "is_final": True, "time_start": time.time(), "time_submit": time.time(),
                           "seed": args.seed}
                tick = time.perf_counter()
                reply = call(proc, owned, args.timeout, request, pcm)
                row[f"{name}_request_seconds"] = time.perf_counter() - tick
                if name == "first":
                    row["first_complete_seconds"] = time.perf_counter() - started
                value = result_fields(reply)
                if expected is None:
                    expected = value
                differences = [key for key in FIELDS if value[key] != expected[key]]
                row[name] = {"result": value, "result_sha256": result_hash(value),
                             "differences": differences, "metrics": reply.get("metrics")}
                stage(f"{name}_complete")
                if differences:
                    raise RuntimeError(f"Recognition differs in fields: {', '.join(differences)}")
            row["passed"] = True
            sampler.finish()
            call(proc, owned, args.timeout, {"command": "shutdown", "id": 4})
            proc.stdin.close()
            proc.wait(timeout=10)
            if proc.returncode != 0:
                raise RuntimeError(f"Service exited with code {proc.returncode}")
    except BaseException as error:
        row["passed"] = False
        row["error"] = repr(error)
    finally:
        try:
            if sampler is not None:
                sampler.finish()
                row["memory_samples"] = sampler.samples
                row["sampling_errors"] = sampler.errors
                row["sampled_peak_totals"] = {
                    key: max((s["totals"][key] for s in sampler.samples), default=0)
                    for key in MEMORY_FIELDS}
                if sampler.errors or not sampler.samples:
                    row["passed"] = False
                    row["error"] = "Memory sampling incomplete; inspect sampling_errors"
            if proc is not None:
                # The job covers descendants too; only handles created by this
                # tool are terminated, never an existing GUI or a name match.
                owned.kill()
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=10)
                deadline = time.monotonic() + 5
                while owned.ids() and time.monotonic() < deadline:
                    time.sleep(0.02)
                row["cleanup_confirmed"] = not owned.ids()
                for stream in (proc.stdin, proc.stdout):
                    if stream and not stream.closed:
                        stream.close()
        finally:
            owned.close()  # KILL_ON_JOB_CLOSE remains a final cleanup backstop.
        if args.profile_load:
            row["load_profiles"] = []
            row["profile_parse_errors"] = []
            log = log_dir / "rust-worker.log"
            row["profile_log"] = str(log)
            if log.is_file():
                for number, line in enumerate(log.read_text("utf-8", errors="replace").splitlines(), 1):
                    if line.startswith("YYUR_LOAD_PROFILE "):
                        try:
                            row["load_profiles"].append(json.loads(line.split(" ", 1)[1]))
                        except json.JSONDecodeError as error:
                            row["profile_parse_errors"].append({"line": number, "error": str(error)})
    return row, expected
