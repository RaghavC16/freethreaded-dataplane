"""Safe JSON object trees plus raw, contiguous CPU tensor bytes."""
from __future__ import annotations
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any, Callable
import torch
from .errors import InvalidObjectError
from .types import ProtocolLimits

DTYPE_BY_NAME = {
    "bool": torch.bool, "uint8": torch.uint8, "int8": torch.int8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "float16": torch.float16, "bfloat16": torch.bfloat16,
    "float32": torch.float32, "float64": torch.float64,
}
NAME_BY_DTYPE = {value: key for key, value in DTYPE_BY_NAME.items()}

@dataclass(frozen=True)
class TypeAdapter:
    """A locally registered, peer-independent custom-object constructor."""
    tag: str
    python_type: type
    encode_fields: Callable[[Any], dict[str, Any]]
    decode_fields: Callable[[dict[str, Any]], Any]

@dataclass(frozen=True)
class EncodedDomainPacket:
    metadata: bytes
    tensor_bytes: tuple[memoryview, ...]
    owners: tuple[Any, ...]
    total_tensor_bytes: int

class DomainPacketCodec:
    def __init__(self, *, limits: ProtocolLimits | None = None,
                 adapters: tuple[TypeAdapter, ...] = ()):
        self.limits = limits or ProtocolLimits()
        self._by_tag: dict[str, TypeAdapter] = {}
        self._by_type: dict[type, TypeAdapter] = {}
        for adapter in adapters:
            self.register_adapter(adapter)

    def register_adapter(self, adapter: TypeAdapter) -> None:
        if not adapter.tag or adapter.tag in self._by_tag:
            raise ValueError(f"duplicate or empty adapter tag: {adapter.tag!r}")
        if adapter.python_type in self._by_type:
            raise ValueError(f"duplicate adapter type: {adapter.python_type!r}")
        self._by_tag[adapter.tag] = adapter
        self._by_type[adapter.python_type] = adapter

    def encode(self, value: Any) -> EncodedDomainPacket:
        descriptors: list[dict[str, Any]] = []
        chunks: list[memoryview] = []
        owners: list[Any] = []
        offset = 0
        elements = 0

        def walk(obj: Any, depth: int) -> Any:
            nonlocal offset, elements
            if depth > self.limits.max_tree_depth:
                raise InvalidObjectError("object tree is too deep")
            elements += 1
            if elements > self.limits.max_container_elements:
                raise InvalidObjectError("object tree has too many elements")
            if obj is None:
                return {"kind": "none"}
            if isinstance(obj, bool):
                return {"kind": "bool", "value": obj}
            if isinstance(obj, int):
                if not -(2**63) <= obj < 2**63:
                    raise InvalidObjectError("integer is outside signed 64-bit range")
                return {"kind": "int", "value": obj}
            if isinstance(obj, float):
                if not math.isfinite(obj):
                    raise InvalidObjectError("non-finite floats are unsupported")
                return {"kind": "float", "value": obj}
            if isinstance(obj, str):
                if len(obj.encode("utf-8")) > self.limits.max_string_bytes:
                    raise InvalidObjectError("string exceeds configured limit")
                return {"kind": "str", "value": obj}
            if isinstance(obj, bytes):
                nbytes = len(obj)
                if offset + nbytes > self.limits.max_tensor_bytes:
                    raise InvalidObjectError("raw region exceeds configured limit")
                index = len(descriptors)
                if index >= self.limits.max_tensor_count:
                    raise InvalidObjectError("too many raw buffers")
                descriptors.append({"kind": "bytes", "offset": offset,
                                    "nbytes": nbytes})
                chunks.append(memoryview(obj))
                owners.append(obj)
                offset += nbytes
                return {"kind": "bytes", "index": index}
            if isinstance(obj, torch.Tensor):
                if obj.layout != torch.strided or obj.dtype not in NAME_BY_DTYPE:
                    raise InvalidObjectError(f"unsupported tensor layout/dtype: {obj.layout}/{obj.dtype}")
                if obj.ndim > self.limits.max_tensor_rank or obj.numel() > self.limits.max_tensor_elements:
                    raise InvalidObjectError("tensor shape exceeds configured limits")
                snapshot = obj.detach().to("cpu").contiguous().clone()
                # Flatten first: PyTorch cannot change element size while viewing a 0-D tensor.
                raw_array = snapshot.reshape(-1).view(torch.uint8).numpy()
                chunk = memoryview(raw_array)
                nbytes = snapshot.numel() * snapshot.element_size()
                if offset + nbytes > self.limits.max_tensor_bytes:
                    raise InvalidObjectError("tensor region exceeds configured limit")
                index = len(descriptors)
                if index >= self.limits.max_tensor_count:
                    raise InvalidObjectError("too many tensors")
                descriptors.append({"kind": "tensor",
                                    "dtype": NAME_BY_DTYPE[snapshot.dtype],
                                    "shape": list(snapshot.shape), "offset": offset,
                                    "nbytes": nbytes, "numel": snapshot.numel()})
                chunks.append(chunk)
                owners.append(snapshot)
                offset += nbytes
                return {"kind": "tensor", "index": index}
            if isinstance(obj, ValueError):
                message = str(obj)
                if len(message.encode("utf-8")) > self.limits.max_string_bytes:
                    raise InvalidObjectError("placeholder message exceeds configured limit")
                return {"kind": "value_error_placeholder", "message": message}
            if isinstance(obj, list):
                return {"kind": "list", "items": [walk(item, depth + 1) for item in obj]}
            if isinstance(obj, tuple):
                return {"kind": "tuple", "items": [walk(item, depth + 1) for item in obj]}
            if isinstance(obj, dict):
                return {"kind": "dict", "items": [[walk(k, depth + 1), walk(v, depth + 1)]
                                                     for k, v in obj.items()]}
            adapter = self._by_type.get(type(obj))
            if adapter is not None:
                fields = adapter.encode_fields(obj)
                if not isinstance(fields, dict) or not all(isinstance(k, str) for k in fields):
                    raise InvalidObjectError("adapter fields must be a string-keyed dict")
                return {"kind": "custom", "tag": adapter.tag,
                        "fields": walk(fields, depth + 1)}
            raise InvalidObjectError(f"unsupported value type: {type(obj).__name__}")

        root = walk(value, 0)
        metadata = json.dumps({"schema": 1, "root": root, "tensors": descriptors},
                              separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(metadata) > self.limits.max_metadata_bytes:
            raise InvalidObjectError("encoded metadata exceeds configured limit")
        return EncodedDomainPacket(metadata, tuple(chunks), tuple(owners), offset)

    def decode(self, metadata: bytes, tensor_bytes: bytes) -> Any:
        if len(metadata) > self.limits.max_metadata_bytes or len(tensor_bytes) > self.limits.max_tensor_bytes:
            raise InvalidObjectError("packet exceeds configured limits")
        try:
            envelope = json.loads(metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise InvalidObjectError("metadata is not valid UTF-8 JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("schema") != 1:
            raise InvalidObjectError("invalid codec schema")
        descriptors = envelope.get("tensors")
        if not isinstance(descriptors, list) or len(descriptors) > self.limits.max_tensor_count:
            raise InvalidObjectError("invalid tensor descriptor table")
        buffers: list[Any] = []
        previous_end = 0
        for desc in descriptors:
            if not isinstance(desc, dict):
                raise InvalidObjectError("invalid raw-buffer descriptor")
            descriptor_kind = desc.get("kind", "tensor")
            if descriptor_kind == "bytes":
                offset, nbytes = desc.get("offset"), desc.get("nbytes")
                if (type(offset) is not int or type(nbytes) is not int
                        or min(offset, nbytes) < 0 or offset != previous_end
                        or offset + nbytes > len(tensor_bytes)):
                    raise InvalidObjectError("invalid bytes descriptor")
                previous_end = offset + nbytes
                buffers.append(bytes(tensor_bytes[offset:previous_end]))
                continue
            if descriptor_kind != "tensor" or desc.get("dtype") not in DTYPE_BY_NAME:
                raise InvalidObjectError("invalid tensor dtype descriptor")
            dtype = DTYPE_BY_NAME[desc["dtype"]]
            shape, offset, nbytes, numel = (desc.get("shape"), desc.get("offset"),
                                            desc.get("nbytes"), desc.get("numel"))
            if (not isinstance(shape, list) or len(shape) > self.limits.max_tensor_rank
                    or not all(type(x) is int and x >= 0 for x in shape)
                    or not all(type(x) is int for x in (offset, nbytes, numel))
                    or min(offset, nbytes, numel) < 0):
                raise InvalidObjectError("invalid tensor shape or range")
            calculated = math.prod(shape)
            expected = calculated * torch.empty((), dtype=dtype).element_size()
            end = offset + nbytes
            if (calculated != numel or expected != nbytes or offset != previous_end
                    or end > len(tensor_bytes) or numel > self.limits.max_tensor_elements):
                raise InvalidObjectError("inconsistent or overlapping tensor descriptor")
            previous_end = end
            if numel == 0:
                tensor = torch.empty(shape, dtype=dtype)
            else:
                owned = bytearray(tensor_bytes[offset:end])
                tensor = torch.frombuffer(owned, dtype=dtype, count=numel).reshape(shape).clone()
            buffers.append(tensor)
        if previous_end != len(tensor_bytes):
            raise InvalidObjectError("tensor region contains unreferenced bytes")
        elements = 0
        def walk(node: Any, depth: int) -> Any:
            nonlocal elements
            if depth > self.limits.max_tree_depth:
                raise InvalidObjectError("object tree is too deep")
            elements += 1
            if elements > self.limits.max_container_elements or not isinstance(node, dict):
                raise InvalidObjectError("malformed object tree")
            kind = node.get("kind")
            if kind == "none": return None
            if kind == "bool" and type(node.get("value")) is bool: return node["value"]
            if kind == "int" and type(node.get("value")) is int and -(2**63) <= node["value"] < 2**63: return node["value"]
            if kind == "float" and type(node.get("value")) in (int, float) and math.isfinite(node["value"]): return float(node["value"])
            if kind == "str" and isinstance(node.get("value"), str) and len(node["value"].encode("utf-8")) <= self.limits.max_string_bytes: return node["value"]
            if kind in ("tensor", "bytes") and type(node.get("index")) is int and 0 <= node["index"] < len(buffers):
                value = buffers[node["index"]]
                if kind == "tensor" and isinstance(value, torch.Tensor): return value
                if kind == "bytes" and isinstance(value, bytes): return value
                raise InvalidObjectError("object kind does not match raw-buffer descriptor")
            if (kind == "value_error_placeholder"
                    and isinstance(node.get("message"), str)
                    and len(node["message"].encode("utf-8")) <= self.limits.max_string_bytes):
                return ValueError(node["message"])
            if kind in ("list", "tuple") and isinstance(node.get("items"), list):
                values = [walk(item, depth + 1) for item in node["items"]]
                return values if kind == "list" else tuple(values)
            if kind == "dict" and isinstance(node.get("items"), list):
                result = {}
                for pair in node["items"]:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise InvalidObjectError("malformed dictionary entry")
                    key, value = walk(pair[0], depth + 1), walk(pair[1], depth + 1)
                    try:
                        if key in result: raise InvalidObjectError("duplicate dict key")
                        result[key] = value
                    except (TypeError, ValueError) as exc: raise InvalidObjectError("unhashable dict key") from exc
                return result
            if kind == "custom" and isinstance(node.get("tag"), str):
                adapter = self._by_tag.get(node["tag"])
                if adapter is None:
                    raise InvalidObjectError(f"unregistered custom tag: {node['tag']}")
                fields = walk(node.get("fields"), depth + 1)
                if not isinstance(fields, dict):
                    raise InvalidObjectError("custom fields are not a dict")
                try: return adapter.decode_fields(fields)
                except Exception as exc: raise InvalidObjectError(f"adapter {adapter.tag} rejected fields") from exc
            raise InvalidObjectError(f"invalid object node kind: {kind!r}")
        return walk(envelope.get("root"), 0)

    def map_tensors(self, value: Any, transform: Callable[[torch.Tensor], torch.Tensor]) -> Any:
        """Rebuild an allowlisted object tree while transforming tensor leaves."""
        if isinstance(value, torch.Tensor): return transform(value)
        if value is None or isinstance(value, (bool, int, float, str, bytes, ValueError)): return value
        if isinstance(value, list): return [self.map_tensors(x, transform) for x in value]
        if isinstance(value, tuple): return tuple(self.map_tensors(x, transform) for x in value)
        if isinstance(value, dict): return {self.map_tensors(k, transform): self.map_tensors(v, transform) for k, v in value.items()}
        adapter = self._by_type.get(type(value))
        if adapter:
            fields = {k: self.map_tensors(v, transform) for k, v in adapter.encode_fields(value).items()}
            return adapter.decode_fields(fields)
        raise InvalidObjectError(f"unsupported value type: {type(value).__name__}")

    def snapshot(self, value: Any) -> Any:
        return self.map_tensors(value, lambda t: t.detach().to("cpu").contiguous().clone())

def dataclass_adapter(python_type: type, tag: str) -> TypeAdapter:
    if not dataclasses.is_dataclass(python_type):
        raise TypeError(f"{python_type!r} is not a dataclass type")
    names = tuple(field.name for field in dataclasses.fields(python_type))
    return TypeAdapter(tag, python_type,
                       lambda obj: {name: getattr(obj, name) for name in names},
                       lambda fields: python_type(**fields))
