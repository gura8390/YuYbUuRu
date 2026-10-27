"""Summarize ORT Node kernel_time provider assignments without starting ORT.

Inputs are paths or globs. Only provider/op/node identity and host timestamps
are retained; tensor data, thread details, transcripts and hotwords are omitted.
Node durations are host profiling intervals, not CUPTI device-kernel timings.
Without --require-heavy-provider, placement acceptance is explicitly unassessed.
"""
from pathlib import Path
import argparse
import glob
import hashlib
import json
import math
import os

HEAVY_OPS = frozenset((
    "Conv", "ConvTranspose", "NhwcConv", "FusedConv", "QLinearConv", "ConvInteger", "MatMul", "Gemm", "FusedMatMul",
    "MatMulInteger", "MatMulNBits", "QLinearMatMul", "DynamicQuantizeMatMul",
    "Gelu", "BiasGelu", "FastGelu", "QuickGelu", "GemmFastGelu", "BiasSplitGelu", "LayerNormalization",
    "SimplifiedLayerNormalization", "SkipLayerNormalization", "SkipSimplifiedLayerNormalization",
    "EmbedLayerNormalization", "GroupNorm", "GroupNormalization", "RMSNorm", "Softmax", "LogSoftmax", "Attention", "MultiHeadAttention",
))
METADATA_OPS = frozenset(("Shape", "Size", "Reshape", "Concat", "Squeeze", "Unsqueeze", "Identity"))
GPU_PROVIDERS = frozenset(("CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider",
                           "ROCMExecutionProvider", "MIGraphXExecutionProvider", "WebGpuExecutionProvider"))
CPU_PROVIDER = "CPUExecutionProvider"
UNKNOWN_PROVIDER = "<missing-provider>"
UNKNOWN_OP = "<missing-op_name>"


def valid_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def interval_union(intervals):
    """Union length on one profile's timestamp axis, including overlapping nodes."""
    if not intervals:
        return None
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def summarize_group(rows):
    nodes = {(row["source_id"], row["node_index"], row["node_name"]) for row in rows}
    intervals = {}
    for row in rows:
        if row["start_us"] is not None and row["duration_us"] is not None:
            intervals.setdefault(row["source_id"], []).append(
                (row["start_us"], row["start_us"] + row["duration_us"]))
    durations = [row["duration_us"] for row in rows if row["duration_us"] is not None]
    per_source = {key: interval_union(value) for key, value in sorted(intervals.items())}
    return {
        "kernel_event_count": len(rows), "unique_node_count": len(nodes),
        "unique_nodes": [{"source_id": source, "node_index": index, "node_name": name}
                         for source, index, name in sorted(nodes, key=lambda item: (item[0], str(item[1]), item[2]))],
        "valid_duration_event_count": len(durations),
        "host_duration_sum_us": sum(durations) if durations else None,
        "host_interval_union_us_by_source": per_source,
        "host_interval_union_us_sum_across_sources": sum(per_source.values()) if per_source else None,
    }


def read_document(document, source_id):
    errors, warnings, selected = [], [], []
    if isinstance(document, list):
        events = document
    elif isinstance(document, dict) and isinstance(document.get("traceEvents"), list):
        events = document["traceEvents"]
    elif isinstance(document, dict) and isinstance(document.get("events"), list):
        events = document["events"]
    else:
        return {"rows": [], "event_count": None, "errors": [{"code": "invalid_event_container"}], "warnings": []}
    if isinstance(document, dict) and any(document.get(key) is False for key in ("complete", "completed")):
        errors.append({"code": "explicitly_incomplete_profile"})
    if not events:
        errors.append({"code": "empty_events"})
    ignored = {"non_node": 0, "node_non_kernel_time": 0, "device_kernel_events": 0}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append({"code": "invalid_event", "event_index": index})
            continue
        if event.get("cat") != "Node":
            ignored["non_node"] += 1
            if event.get("cat") == "Kernel":
                ignored["device_kernel_events"] += 1
            continue
        name = event.get("name")
        if not isinstance(name, str) or not (name == "kernel_time" or name.endswith("_kernel_time")):
            ignored["node_non_kernel_time"] += 1
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        provider = args.get("provider")
        op = args.get("op_name")
        if not isinstance(provider, str) or not provider.strip():
            provider = UNKNOWN_PROVIDER
            errors.append({"code": "missing_provider", "event_index": index})
        else:
            provider = provider.strip()
        if not isinstance(op, str) or not op.strip():
            op = UNKNOWN_OP
            errors.append({"code": "missing_op_name", "event_index": index})
        else:
            op = op.strip()
        duration = event.get("dur")
        timestamp = event.get("ts")
        if event.get("ph") != "X":
            errors.append({"code": "incomplete_node_interval", "event_index": index})
            duration = None
        elif not valid_number(duration):
            errors.append({"code": "invalid_or_missing_duration", "event_index": index})
            duration = None
        if not valid_number(timestamp):
            errors.append({"code": "invalid_or_missing_timestamp", "event_index": index})
            timestamp = None
        if duration is not None and timestamp is not None and not valid_number(timestamp + duration):
            errors.append({"code": "interval_overflow", "event_index": index})
            timestamp = None
        node_index = args.get("node_index")
        if node_index is not None and (isinstance(node_index, bool) or not isinstance(node_index, (str, int))):
            errors.append({"code": "invalid_node_index", "event_index": index})
            node_index = None
        suffix = "_kernel_time" if name.endswith("_kernel_time") else "kernel_time"
        node_name = name[:-len(suffix)] or "<unnamed>"
        selected.append({"source_id": source_id, "event_index": index, "provider": provider, "op_name": op,
                         "base_op_name": op.rsplit("::", 1)[-1], "node_name": node_name,
                         "node_index": str(node_index) if node_index is not None else None,
                         "duration_us": duration, "start_us": timestamp})
    if not selected:
        errors.append({"code": "no_node_kernel_time_events"})
    if ignored["device_kernel_events"]:
        warnings.append({"code": "device_kernel_events_not_summarized",
                         "count": ignored["device_kernel_events"]})
    return {"rows": selected, "event_count": len(events), "ignored": ignored, "errors": errors, "warnings": warnings}


def summarize_profiles(paths, required_heavy_provider=None, input_errors=()):
    sources, all_rows, errors, warnings = [], [], list(input_errors), []
    for source_index, path in enumerate(paths):
        path = Path(path)
        source_id = f"profile-{source_index:03}"
        source = {"id": source_id, "path": str(path), "sha256": None, "bytes": None,
                  "data_complete": False, "errors": [], "warnings": []}
        sources.append(source)
        try:
            raw = path.read_bytes()
            source.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            document = json.loads(raw.decode("utf-8-sig"))
        except json.JSONDecodeError as error:
            source["errors"].append({"code": "invalid_or_truncated_json", "line": error.lineno, "column": error.colno})
            continue
        except (OSError, UnicodeError) as error:
            source["errors"].append({"code": type(error).__name__})
            continue
        parsed = read_document(document, source_id)
        source.update(event_count=parsed["event_count"], ignored=parsed.get("ignored"),
                      errors=parsed["errors"], warnings=parsed["warnings"],
                      data_complete=not parsed["errors"], **summarize_group(parsed["rows"]))
        all_rows.extend(parsed["rows"])
    if not sources:
        errors.append({"code": "no_profile_inputs"})
    for source in sources:
        errors.extend({"source_id": source["id"], **error} for error in source["errors"])
        warnings.extend({"source_id": source["id"], **warning} for warning in source["warnings"])
    providers = {}
    for provider in sorted({row["provider"] for row in all_rows}):
        rows = [row for row in all_rows if row["provider"] == provider]
        providers[provider] = {**summarize_group(rows),
            "ops": {op: summarize_group([row for row in rows if row["op_name"] == op])
                    for op in sorted({row["op_name"] for row in rows})}}
    heavy = {}
    metadata = {}
    for base_op in sorted({row["base_op_name"] for row in all_rows}):
        rows = [row for row in all_rows if row["base_op_name"] == base_op]
        if base_op not in HEAVY_OPS and base_op not in METADATA_OPS:
            continue
        entry = {"providers": {provider: summarize_group([row for row in rows if row["provider"] == provider])
                               for provider in sorted({row["provider"] for row in rows})},
                 "cpu_event_count": sum(row["provider"] == CPU_PROVIDER for row in rows),
                 "gpu_provider_event_count": sum(row["provider"] in GPU_PROVIDERS for row in rows),
                 "other_or_missing_provider_event_count": sum(row["provider"] != CPU_PROVIDER and
                     row["provider"] not in GPU_PROVIDERS for row in rows)}
        (heavy if base_op in HEAVY_OPS else metadata)[base_op] = entry
    heavy_rows = [row for row in all_rows if row["base_op_name"] in HEAVY_OPS]
    cpu_other = [row for row in all_rows if row["provider"] == CPU_PROVIDER and
                 row["base_op_name"] not in HEAVY_OPS | METADATA_OPS]
    data_complete = bool(sources) and not errors
    placement = {"evaluated": required_heavy_provider is not None, "passed": None,
                 "required_heavy_provider": required_heavy_provider,
                 "observed_heavy_event_count": len(heavy_rows), "violating_heavy_events": None,
                 "scope": "Node provider assignment only; no device-kernel timing or recognition-quality acceptance."}
    if required_heavy_provider is not None:
        violations = [row for row in heavy_rows if row["provider"] != required_heavy_provider]
        placement["violating_heavy_events"] = [{key: row[key] for key in
            ("source_id", "event_index", "node_name", "node_index", "op_name", "provider")} for row in violations]
        placement["passed"] = data_complete and bool(heavy_rows) and not violations
    return {"schema": 1, "data_complete": data_complete, "sources": sources, "providers": providers,
            "all_node_kernel_time": summarize_group(all_rows), "heavy_ops": heavy,
            "cpu_allowed_metadata_ops": metadata,
            "other_cpu_ops": {op: summarize_group([row for row in cpu_other if row["op_name"] == op])
                              for op in sorted({row["op_name"] for row in cpu_other})},
            "placement_acceptance": placement, "errors": errors, "warnings": warnings,
            "method": {"events": "cat=Node, name ends kernel_time, complete ph=X intervals; args.provider/op_name.",
                       "time_unit": "microseconds as recorded by ORT/Chrome trace",
                       "duration": "Host Node intervals only; not CUPTI device-kernel time. CUDA provider presence alone is not acceptance.",
                       "sum_vs_union": "Sum may double count overlapping host intervals. Union is computed per profile; independent timestamp origins are never merged across files.",
                       "unique_node_identity": "Profile ID + args.node_index (if present) + Node name without kernel_time suffix.",
                       "capture_boundary": "A valid JSON trace does not prove all intended inference requests finished; caller must verify its run receipt.",
                       "metadata_cpu_policy": "Shape/reshape/concat and listed metadata ops may remain on CPU, but their cost is still reported; this classification does not prove they are cheap.",
                       "unclassified_ops": "All providers/op names are listed; unclassified/custom/fused ops are not silently treated as heavy-GPU proof."},
            "heavy_op_classification": sorted(HEAVY_OPS), "metadata_op_classification": sorted(METADATA_OPS),
            "inference_completion_verified": False, "transcript_text_in_report": False,
            "documentation": "https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html"}


def expand_inputs(patterns):
    paths, errors, seen = [], [], set()
    for pattern in patterns:
        matches = [Path(pattern)] if Path(pattern).is_file() else [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
        matches = [path for path in matches if path.is_file()]
        if not matches:
            errors.append({"code": "input_path_or_glob_unmatched", "input": pattern})
        for path in matches:
            path = path.resolve()
            key = os.path.normcase(str(path))
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", help="One or more ORT JSON paths or quoted globs")
    parser.add_argument("--output", type=Path, help="Write JSON instead of stdout; refuses to overwrite")
    parser.add_argument("--require-heavy-provider", help="Optional strict provider-assignment gate for every observed classified heavy node")
    args = parser.parse_args()
    paths, errors = expand_inputs(args.profiles)
    report = summarize_profiles(paths, args.require_heavy_provider, errors)
    value = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(value)
    else:
        print(value, end="")
    if not report["data_complete"] or report["placement_acceptance"]["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
