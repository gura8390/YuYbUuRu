"""Read owned-process modules and WDDM GPU memory; never initialize a model.

Collector().snapshot(pids) returns JSON-safe data. Missing GPU rows/unsupported
counters have null values and explicit errors; device-global VRAM is never used
as a substitute. Module paths are local evidence and must not be published.
The caller owns PID authorization/lifetime checks. This module opens no apps.
"""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes as w
import json
import ntpath
import os
import re
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PDH_MORE_DATA = 0x800007D2
PDH_FMT_LARGE = 0x00000400
PDH_FMT_NOSCALE = 0x00001000
VALID_CSTATUS = (0, 1)  # PDH_CSTATUS_VALID_DATA / PDH_CSTATUS_NEW_DATA
MAX_ARRAY_BYTES = 16 * 1024 * 1024
COUNTERS = {
    "dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
    "shared_bytes": r"\GPU Process Memory(*)\Shared Usage",
}
INSTANCE = re.compile(
    r"^pid_(?P<pid>\d+)_luid_(?P<high>0x[0-9a-f]+)_(?P<low>0x[0-9a-f]+)"
    r"_phys_(?P<physical>\d+)(?:#(?P<duplicate>\d+))?$", re.I)
RELEVANT = ("llama", "ggml", "cudart", "cublas", "cudnn", "cufft", "curand", "cusolver",
            "cusparse", "nvcuda", "nvml", "nvrtc", "onnxruntime", "directml", "vulkan-", "libomp")


class _ValueUnion(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_int32), ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_int64), ("stringValue", ctypes.c_void_p)]


class _CounterValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("CStatus", ctypes.c_uint32), ("value", _ValueUnion)]


class _CounterItem(ctypes.Structure):
    # A raw pointer permits checking the returned name lies inside the array
    # allocation before decoding its UTF-16 string.
    _fields_ = [("szName", ctypes.c_void_p), ("FmtValue", _CounterValue)]


def _u32(value):
    return int(value) & 0xFFFFFFFF


def _error(operation, code=None, message=None):
    result = {"operation": operation, "code": f"0x{_u32(code):08X}" if code is not None else None}
    if message is not None:
        result["message"] = str(message)
    return result


def parse_instance(name):
    """Decode a GPU Process Memory instance without guessing a PID or adapter."""
    match = INSTANCE.fullmatch(name)
    if not match:
        return None
    return {"pid": int(match["pid"]),
            "adapter_luid": f"{match['high'].lower()}_{match['low'].lower()}",
            "physical_adapter_index": int(match["physical"]),
            "duplicate_instance_index": int(match["duplicate"]) if match["duplicate"] else None}


class Collector:
    """Reusable, serialized PDH query plus read-only process-module enumeration.

    Use at phase boundaries or a measured low sampling rate: enumerating every
    loaded module is more expensive than a host-memory counter read. close()
    returns a cleanup receipt and is idempotent. No subprocesses are created.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._closed = False
        self._query = ctypes.c_void_p()
        self._counters = {}
        self._counter_errors = {}
        self._gpu_errors = []
        self._module_error = None
        self._kernel = self._psapi = self._pdh = None
        self._close_receipt = None
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            failure = _error("platform", message="64-bit Windows Python is required")
            self._module_error = failure
            self._gpu_errors.append(failure)
            return
        try:
            self._kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
            self._psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
            self._kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
            self._kernel.OpenProcess.restype = w.HANDLE
            self._kernel.CloseHandle.argtypes = [w.HANDLE]
            self._kernel.CloseHandle.restype = w.BOOL
            self._kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
            self._kernel.GetProcessTimes.restype = w.BOOL
            self._psapi.EnumProcessModulesEx.argtypes = [w.HANDLE, ctypes.POINTER(ctypes.c_void_p),
                                                       w.DWORD, ctypes.POINTER(w.DWORD), w.DWORD]
            self._psapi.EnumProcessModulesEx.restype = w.BOOL
            self._psapi.GetModuleFileNameExW.argtypes = [w.HANDLE, ctypes.c_void_p, w.LPWSTR, w.DWORD]
            self._psapi.GetModuleFileNameExW.restype = w.DWORD
        except (OSError, AttributeError) as error:
            self._module_error = _error("module_api_init", message=error)
        try:
            self._pdh = ctypes.WinDLL("pdh.dll", use_last_error=True)
            self._pdh.PdhOpenQueryW.argtypes = [w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            self._pdh.PdhOpenQueryW.restype = ctypes.c_int32
            self._pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, w.LPCWSTR, ctypes.c_size_t,
                                                       ctypes.POINTER(ctypes.c_void_p)]
            self._pdh.PdhAddEnglishCounterW.restype = ctypes.c_int32
            self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            self._pdh.PdhCollectQueryData.restype = ctypes.c_int32
            self._pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, w.DWORD,
                ctypes.POINTER(w.DWORD), ctypes.POINTER(w.DWORD), ctypes.c_void_p]
            self._pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_int32
            self._pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            self._pdh.PdhCloseQuery.restype = ctypes.c_int32
            status = _u32(self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)))
            if status:
                self._gpu_errors.append(_error("PdhOpenQueryW", status))
                return
            for key, path in COUNTERS.items():
                counter = ctypes.c_void_p()
                status = _u32(self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(counter)))
                if status:
                    self._counter_errors[key] = _error("PdhAddEnglishCounterW", status)
                else:
                    self._counters[key] = counter
            # Prime once. GPU memory gauges generally need only one collection;
            # snapshot still collects anew and never reuses stale values.
            if self._counters:
                self._pdh.PdhCollectQueryData(self._query)
        except (OSError, AttributeError) as error:
            self._gpu_errors.append(_error("pdh_api_init", message=error))

    def _modules(self, pid):
        row = {"pid": pid, "process_creation_time_100ns": None, "complete": False,
               "modules": None, "relevant_modules": None, "errors": []}
        if self._module_error:
            row["errors"].append(self._module_error)
            return row
        handle = self._kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            row["errors"].append(_error("OpenProcess", ctypes.get_last_error()))
            return row
        try:
            times = [w.FILETIME() for _ in range(4)]
            if self._kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
                row["process_creation_time_100ns"] = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            else:
                row["errors"].append(_error("GetProcessTimes", ctypes.get_last_error()))
            capacity = 256
            for _ in range(4):
                modules = (ctypes.c_void_p * capacity)()
                needed = w.DWORD()
                if not self._psapi.EnumProcessModulesEx(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
                    row["errors"].append(_error("EnumProcessModulesEx", ctypes.get_last_error()))
                    return row
                if needed.value <= ctypes.sizeof(modules):
                    break
                capacity = needed.value // ctypes.sizeof(ctypes.c_void_p) + 64
                if capacity > 16384:
                    row["errors"].append(_error("EnumProcessModulesEx", message="Module count exceeds safety bound"))
                    return row
            else:
                row["errors"].append(_error("EnumProcessModulesEx", message="Module list kept growing"))
                return row
            if not needed.value or needed.value % ctypes.sizeof(ctypes.c_void_p):
                row["errors"].append(_error("EnumProcessModulesEx", message="Unaligned module-array byte count"))
                return row
            found = []
            for module in modules[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
                for chars in (1024, 32768):
                    buffer = ctypes.create_unicode_buffer(chars)
                    length = self._psapi.GetModuleFileNameExW(handle, module, buffer, chars)
                    if not length:
                        row["errors"].append(_error("GetModuleFileNameExW", ctypes.get_last_error()))
                        break
                    if length < chars - 1:
                        path = buffer.value
                        found.append({"basename": ntpath.basename(path), "path": path})
                        break
                else:
                    row["errors"].append(_error("GetModuleFileNameExW", message="Module path was truncated"))
            row["modules"] = sorted(found, key=lambda item: item["path"].lower())
            row["relevant_modules"] = [item for item in row["modules"]
                                       if item["basename"].lower().startswith(RELEVANT)]
            row["complete"] = not row["errors"]
            return row
        finally:
            if not self._kernel.CloseHandle(handle):
                row["errors"].append(_error("CloseHandle", ctypes.get_last_error()))
                row["complete"] = False

    def _array(self, counter):
        """Restart the size query after MORE_DATA; a short data buffer's size is unreliable."""
        attempts = []
        for attempt in range(4):
            size, count = w.DWORD(), w.DWORD()
            status = _u32(self._pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE | PDH_FMT_NOSCALE,
                ctypes.byref(size), ctypes.byref(count), None))
            attempts.append({"phase": "size", "status": f"0x{status:08X}", "bytes": size.value,
                             "item_count": count.value})
            if status == 0 and size.value == 0 and count.value == 0:
                return [], [], attempts
            if status != PDH_MORE_DATA:
                return None, [_error("PdhGetFormattedCounterArrayW:size", status)], attempts
            if not ctypes.sizeof(_CounterItem) <= size.value <= MAX_ARRAY_BYTES:
                return None, [_error("PdhGetFormattedCounterArrayW:size", message="Invalid/bounded array size")], attempts
            allocated = size.value
            buffer = ctypes.create_string_buffer(allocated)
            status = _u32(self._pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE | PDH_FMT_NOSCALE,
                ctypes.byref(size), ctypes.byref(count), buffer))
            attempts.append({"phase": "data", "status": f"0x{status:08X}", "bytes": size.value,
                             "item_count": count.value})
            if status == PDH_MORE_DATA:
                continue
            if status:
                return None, [_error("PdhGetFormattedCounterArrayW:data", status)], attempts
            if size.value > allocated or count.value * ctypes.sizeof(_CounterItem) > allocated:
                return None, [_error("PdhGetFormattedCounterArrayW:data", message="Invalid array bounds")], attempts
            begin, end = ctypes.addressof(buffer), ctypes.addressof(buffer) + allocated
            items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
            rows, errors = [], []
            for index in range(count.value):
                item = items[index]
                pointer = item.szName or 0
                if not begin <= pointer < end:
                    errors.append(_error("PDH item name", message="Name pointer outside array allocation"))
                    continue
                chars = min(1024, (end - pointer) // ctypes.sizeof(ctypes.c_wchar))
                name = ctypes.wstring_at(pointer, chars)
                if "\0" not in name:
                    errors.append(_error("PDH item name", message="Unterminated instance name"))
                    continue
                name = name.split("\0", 1)[0]
                cstatus = _u32(item.FmtValue.CStatus)
                valid = cstatus in VALID_CSTATUS and item.FmtValue.largeValue >= 0
                rows.append({"instance": name, "cstatus": f"0x{cstatus:08X}",
                             "value": int(item.FmtValue.largeValue) if valid else None,
                             "error": None if valid else _error("PDH CStatus/value", cstatus,
                                                               "Invalid counter data; value unavailable")})
            return rows, errors, attempts
        return None, [_error("PdhGetFormattedCounterArrayW", PDH_MORE_DATA, "Instance array changed repeatedly")], attempts

    def _gpu(self, pids):
        result = {"source": "WDDM GPU Process Memory / PdhAddEnglishCounterW",
                  "rows": [], "counters": {}, "pid_coverage": [], "errors": list(self._gpu_errors),
                  "sampling_ok": False, "global_vram_substitution": False}
        collect_status = None
        if self._query.value and self._counters and self._pdh:
            collect_status = _u32(self._pdh.PdhCollectQueryData(self._query))
            if collect_status:
                result["errors"].append(_error("PdhCollectQueryData", collect_status))
        merged = {}
        for key, path in COUNTERS.items():
            info = {"path": path, "available": False, "array_item_count": None,
                    "matched_item_count": None, "attempts": [], "errors": []}
            result["counters"][key] = info
            if key in self._counter_errors:
                info["errors"].append(self._counter_errors[key])
                continue
            if key not in self._counters or collect_status is None or collect_status:
                info["errors"].append(_error("counter unavailable", message="No fresh successful query collection"))
                continue
            rows, errors, attempts = self._array(self._counters[key])
            info["attempts"], info["errors"] = attempts, errors
            if rows is None:
                continue
            info["available"] = not errors
            info["array_item_count"] = len(rows)
            info["matched_item_count"] = 0
            for item in rows:
                parsed = parse_instance(item["instance"])
                if parsed is None:
                    if any(item["instance"].lower().startswith(f"pid_{pid}_") for pid in pids):
                        info["errors"].append(_error("parse GPU instance", message="Owned PID has an unrecognized instance"))
                        info["available"] = False
                    continue
                if parsed["pid"] not in pids:
                    continue
                info["matched_item_count"] += 1
                row = merged.setdefault(item["instance"].lower(), {**parsed, "instance": item["instance"],
                    "dedicated_bytes": None, "shared_bytes": None, "cstatus": {}, "errors": []})
                if key in row["cstatus"]:
                    row[key] = None
                    row["errors"].append(_error("duplicate GPU counter row", message=key))
                else:
                    row[key] = item["value"]
                    row["cstatus"][key] = item["cstatus"]
                    if item["error"]:
                        row["errors"].append(item["error"])
        result["rows"] = sorted(merged.values(), key=lambda row: (row["pid"], row["adapter_luid"],
                                                                  row["physical_adapter_index"], row["instance"]))
        for row in result["rows"]:
            for key in COUNTERS:
                if key not in row["cstatus"]:
                    row["errors"].append(_error("missing GPU counter", message=key))
        for pid in pids:
            rows = [row for row in result["rows"] if row["pid"] == pid]
            errors = []
            if not rows:
                errors.append(_error("no matching GPU instance", message="No PID/LUID row; not measured zero"))
            elif len(rows) > 1:
                errors.append(_error("multiple GPU instances", message="No cross-adapter/instance aggregate is inferred"))
            elif any(row["errors"] for row in rows):
                errors.append(_error("incomplete GPU instance", message="Inspect per-row errors/null values"))
            result["pid_coverage"].append({"pid": pid, "instance_count": len(rows),
                "status": "ok" if not errors else ("partial" if rows else "no_matching_instance"),
                "dedicated_bytes": rows[0]["dedicated_bytes"] if len(rows) == 1 else None,
                "shared_bytes": rows[0]["shared_bytes"] if len(rows) == 1 else None,
                "aggregation": "Single instance copied; multiple adapters/instances remain unaggregated", "errors": errors})
        result["sampling_ok"] = not result["errors"] and all(
            info["available"] and not info["errors"] for info in result["counters"].values()) and not any(
                row["errors"] for row in result["rows"])
        return result

    def snapshot(self, pids):
        values = list(pids)
        if any(not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid <= 0xFFFFFFFF for pid in values):
            raise ValueError("PIDs must be positive DWORD integers")
        values = sorted(set(values))
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("Collector is closed")
            modules = [self._modules(pid) for pid in values]
            gpu = self._gpu(values)
        return {"schema": 1, "pids": values, "perf_counter_seconds": started,
                "wall_time_ns": time.time_ns(), "collector_seconds": time.perf_counter() - started,
                "process_modules": modules, "gpu_process_memory": gpu,
                "scope": "Owned PIDs only; module paths are local/private; GPU values are PDH reported bytes, not host commit or device-global VRAM.",
                "module_list_can_change_during_snapshot": True}

    def close(self):
        with self._lock:
            if self._closed:
                return self._close_receipt
            self._closed = True
            status = _u32(self._pdh.PdhCloseQuery(self._query)) if self._query.value and self._pdh else 0
            self._query = ctypes.c_void_p()
            self._counters.clear()
            self._close_receipt = {"closed": status == 0, "error": _error("PdhCloseQuery", status) if status else None}
            return self._close_receipt

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


collector = Collector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pids", type=int, nargs="+", help="Authorized owned PIDs; default is this Python process")
    parser.add_argument("--output", type=Path, help="Local JSON under this checkout's evidence/private")
    args = parser.parse_args()
    pids = args.pids if args.pids is not None else [os.getpid()]
    output = (args.output or ROOT / "evidence/private" / f"cuda-metrics-{time.time_ns()}.json").resolve()
    if not output.is_relative_to((ROOT / "evidence/private").resolve()):
        parser.error("Output must stay under this checkout's evidence/private")
    if output.exists():
        parser.error("Refusing to overwrite metrics evidence")
    sampler = Collector()
    try:
        report = sampler.snapshot(pids)
    finally:
        closed = sampler.close()
    report["cleanup"] = closed
    report["model_started"] = False
    report["apps_started"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "pid_count": len(pids),
                      "module_enumeration_complete": all(row["complete"] for row in report["process_modules"]),
                      "gpu_rows": len(report["gpu_process_memory"]["rows"]),
                      "gpu_sampling_ok": report["gpu_process_memory"]["sampling_ok"], "closed": closed["closed"]}))


if __name__ == "__main__":
    main()
