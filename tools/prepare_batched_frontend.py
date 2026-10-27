"""Prepare an isolated dynamic-batch Qwen3-ASR frontend, without inference.

Only the two public batch dimensions and the Reshape batch constant change.
All operation nodes, attributes, learned weights and other constants retain
their exact protobuf bytes. This tool uses no ONNX/ORT dependency or session.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
PINNED_SOURCE_SHA256 = "8c29d210b934f377983c0daa28d276f36cbc9a8a823d59d3afc5da2fd8eda2f6"


@dataclass
class Field:
    number: int
    wire: int
    value: object
    raw: memoryview


def read_varint(data, position):
    value = shift = 0
    while position < len(data) and shift < 70:
        item = data[position]
        position += 1
        value |= (item & 127) << shift
        if item < 128:
            return value, position
        shift += 7
    raise ValueError("Truncated or invalid protobuf varint")


def varint(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def fields(data):
    data = memoryview(data)
    position = 0
    while position < len(data):
        start = position
        tag, position = read_varint(data, position)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("Invalid protobuf field zero")
        if wire == 0:
            value, position = read_varint(data, position)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[position:position + size]
            position += size
        elif wire == 2:
            size, position = read_varint(data, position)
            value = data[position:position + size]
            position += size
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if position > len(data):
            raise ValueError("Truncated protobuf field")
        yield Field(number, wire, value, data[start:position])


def text(value):
    return bytes(value).decode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def length_field(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def unique(data, number, wire=2):
    matches = [field for field in fields(data) if field.number == number]
    if len(matches) != 1 or matches[0].wire != wire:
        raise ValueError(f"Expected one protobuf field {number} with wire {wire}")
    return matches[0]


def replace(data, number, callback):
    target = unique(data, number)
    return b"".join(length_field(number, callback(field.value)) if field.number == number
                    else bytes(field.raw) for field in fields(data))


def name(data, number=1):
    return text(unique(data, number).value)


def dimensions(value_info):
    tensor = unique(unique(value_info, 2).value, 1).value
    dtype = unique(tensor, 1, 0).value
    shape = unique(tensor, 2).value
    result = []
    for dimension in fields(shape):
        if dimension.number != 1:
            raise ValueError("Unexpected tensor-shape field")
        dim = list(fields(dimension.value))
        if len(dim) != 1 or dim[0].number not in (1, 2):
            raise ValueError("Unexpected dimension metadata")
        result.append(dim[0].value if dim[0].number == 1 else text(dim[0].value))
    return dtype, result


def dynamic_batch(value_info):
    def shape_update(shape):
        dimensions = list(fields(shape))
        return length_field(1, length_field(2, b"batch")) + b"".join(bytes(dim.raw) for dim in dimensions[1:])
    return replace(value_info, 2, lambda tp: replace(tp, 1, lambda tensor: replace(tensor, 2, shape_update)))


def tensor_info(tensor):
    dims = []
    for field in fields(tensor):
        if field.number == 1:
            if field.wire == 0:
                dims.append(field.value)
            elif field.wire == 2:
                position = 0
                while position < len(field.value):
                    value, position = read_varint(field.value, position)
                    dims.append(value)
            else:
                raise ValueError("Unexpected TensorProto dims encoding")
        if field.number == 13 or (field.number == 14 and field.value != 0):
            raise ValueError("Only inline initializers are accepted")
    return {"name": name(tensor, 8), "dims": dims, "dtype": unique(tensor, 2, 0).value,
            "protobuf_sha256": digest(tensor)}


def graph_view(graph):
    initializers = {}
    inputs, outputs, nodes = [], [], []
    for field in fields(graph):
        if field.number == 5:
            info = tensor_info(field.value)
            if info["name"] in initializers:
                raise ValueError("Duplicate initializer")
            initializers[info["name"]] = (field, info)
        elif field.number in (11, 12):
            (inputs if field.number == 11 else outputs).append(field)
        elif field.number == 1:
            nodes.append(field)
    return initializers, inputs, outputs, nodes


def grouped_hash(initializers, excluded):
    result = hashlib.sha256()
    for key in sorted(set(initializers) - set(excluded)):
        value = initializers[key][0].value
        label = key.encode("utf-8")
        result.update(struct.pack("<Q", len(label)))
        result.update(label)
        result.update(struct.pack("<Q", len(value)))
        result.update(value)
    return result.hexdigest()


def prepare(source):
    if digest(source) != PINNED_SOURCE_SHA256:
        raise ValueError("Source is not the pinned accepted int4 frontend; no version stamp or output created")
    graph_field = unique(source, 7)
    graph = graph_field.value
    initializers, inputs, outputs, nodes = graph_view(graph)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one frontend input/output")
    if name(inputs[0].value) != "chunk_mel" or dimensions(inputs[0].value) != (1, [1, 128, 100]):
        raise ValueError("Unexpected frontend input metadata")
    if name(outputs[0].value) != "chunk_out" or dimensions(outputs[0].value) != (1, [1, 13, 896]):
        raise ValueError("Unexpected frontend output metadata")
    operations = Counter(name(node.value, 4) for node in nodes)
    if operations != {"Unsqueeze": 1, "Conv": 3, "Gelu": 3, "Transpose": 1, "Reshape": 1, "MatMulNBits": 1, "Add": 1}:
        raise ValueError("Unexpected frontend operation topology")
    control_name = "val_26"
    shape_tensor = initializers[control_name][0].value
    if initializers[control_name][1]["dims"] != [3] or initializers[control_name][1]["dtype"] != 7:
        raise ValueError("Unexpected Reshape constant type or shape")
    if bytes(unique(shape_tensor, 9).value) != struct.pack("<3q", 1, 13, 7680):
        raise ValueError("Unexpected Reshape constant values")
    # allowzero remains 1. A zero here would mean a zero-length dimension;
    # -1 infers batch and preserves every node's original attributes.
    changed_shape = replace(shape_tensor, 9, lambda _: struct.pack("<3q", -1, 13, 7680))
    pieces = []
    for field in fields(graph):
        if field.number == 11:
            pieces.append(length_field(11, dynamic_batch(field.value)))
        elif field.number == 12:
            pieces.append(length_field(12, dynamic_batch(field.value)))
        elif field.number == 5 and name(field.value, 8) == control_name:
            pieces.append(length_field(5, changed_shape))
        else:
            pieces.append(bytes(field.raw))
    new_graph = b"".join(pieces)
    candidate = replace(source, 7, lambda _: new_graph)
    new_initializers, new_inputs, new_outputs, new_nodes = graph_view(new_graph)
    assert dimensions(new_inputs[0].value) == (1, ["batch", 128, 100])
    assert dimensions(new_outputs[0].value) == (1, ["batch", 13, 896])
    assert [bytes(node.raw) for node in nodes] == [bytes(node.raw) for node in new_nodes]
    assert set(initializers) == set(new_initializers)
    unchanged = []
    for key in initializers:
        if key != control_name:
            assert bytes(initializers[key][0].raw) == bytes(new_initializers[key][0].raw), key
            unchanged.append(initializers[key][1])
    old_other = [bytes(field.raw) for field in fields(graph) if field.number not in (11, 12)
                 and not (field.number == 5 and name(field.value, 8) == control_name)]
    new_other = [bytes(field.raw) for field in fields(new_graph) if field.number not in (11, 12)
                 and not (field.number == 5 and name(field.value, 8) == control_name)]
    assert old_other == new_other
    assert [bytes(f.raw) for f in fields(source) if f.number != 7] == [bytes(f.raw) for f in fields(candidate) if f.number != 7]
    old_hash, new_hash = grouped_hash(initializers, [control_name]), grouped_hash(new_initializers, [control_name])
    assert old_hash == new_hash
    return candidate, {"schema": 1, "prepared": True, "inference_calls": 0,
        "source_sha256": digest(source), "candidate_sha256": digest(candidate),
        "source_weights_sha256": old_hash, "candidate_weights_sha256": new_hash,
        "weights_hash_scope": "Sorted named TensorProto bytes for all immutable weights/constants, excluding only val_26 shape control",
        "source_input": [1, 128, 100], "candidate_input": ["batch", 128, 100],
        "source_output": [1, 13, 896], "candidate_output": ["batch", 13, 896],
        "shape_control": {"name": control_name, "source": [1, 13, 7680], "candidate": [-1, 13, 7680]},
        "all_nodes_and_attributes_byte_identical": True, "other_model_and_graph_fields_byte_identical": True,
        "unchanged_initializers": unchanged, "node_count": len(nodes),
        "quality_equivalence": "Not established by graph rewriting; requires full float32 bit comparison"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .onnx file inside this project's evidence/ or .build/")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    receipt_path = output.with_suffix(".receipt.json")
    if not any(output.is_relative_to(ROOT / area) for area in ["evidence", ".build"]):
        raise ValueError("Experimental models must stay in this project's ignored evidence/ or .build/ directory")
    if source == output or output.exists() or receipt_path.exists():
        raise ValueError("Refusing to overwrite the source, existing output or receipt")
    original = source.read_bytes()
    candidate, receipt = prepare(original)
    receipt.update({"source": str(source), "candidate": str(output)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(candidate)
    assert digest(output.read_bytes()) == receipt["candidate_sha256"]
    assert digest(source.read_bytes()) == receipt["source_sha256"]
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"prepared": True, "output": str(output), "receipt": str(receipt_path),
                      "weights_identical": True, "inference_calls": 0, "candidate_sha256": receipt["candidate_sha256"]}))


if __name__ == "__main__":
    main()
