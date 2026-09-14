from __future__ import annotations
import json
import unittest
from dataclasses import dataclass
import torch
from nogil_data_plane import (
    DomainPacketCodec,
    DomainRpcSerializer,
    InvalidObjectError,
    ProtocolLimits,
    dataclass_adapter,
)
from nogil_data_plane.types import SharedDomainListState

@dataclass
class CustomDomain:
    values: dict
    count: int

class ProtocolCodecTests(unittest.TestCase):
    def test_round_trip_nested_tensors_and_adapter(self):
        codec = DomainPacketCodec(adapters=(dataclass_adapter(CustomDomain, "test.custom"),))
        source_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        source = {1: CustomDomain({"tensor": source_tensor, "empty": torch.empty(0),
                                  "scalar": torch.tensor(7, dtype=torch.int64)}, 2),
                  "tuple": (True, None, ValueError("known placeholder"))}
        packet = codec.encode(source)
        decoded = codec.decode(packet.metadata, b"".join(packet.tensor_bytes))
        self.assertIsInstance(decoded[1], CustomDomain)
        self.assertTrue(torch.equal(decoded[1].values["tensor"], source_tensor))
        self.assertEqual(decoded[1].values["empty"].numel(), 0)
        self.assertEqual(decoded[1].values["scalar"].shape, torch.Size([]))
        self.assertEqual(decoded[1].values["scalar"].item(), 7)
        self.assertIsInstance(decoded["tuple"][2], ValueError)
        source_tensor.fill_(99)
        self.assertFalse(torch.all(decoded[1].values["tensor"] == 99))

    def test_round_trip_expanded_and_zero_stride_tensors(self):
        codec = DomainPacketCodec()
        expanded = torch.tensor(3, dtype=torch.int32).expand(1)
        broadcast = torch.tensor([[1.5]]).expand(4, 3)
        self.assertEqual(expanded.stride(), (0,))
        packet = codec.encode({"expanded": expanded, "broadcast": broadcast})
        decoded = codec.decode(packet.metadata, b"".join(packet.tensor_bytes))
        self.assertTrue(torch.equal(decoded["expanded"], expanded))
        self.assertTrue(torch.equal(decoded["broadcast"], broadcast))
        self.assertEqual(codec.snapshot(expanded).stride(), (1,))

    def test_round_trip_raw_bytes_and_rpc_envelope(self):
        serializer = DomainRpcSerializer(DomainPacketCodec())
        source = {
            "bootstrap": b"\x00domain-list\xff",
            "tensor": torch.tensor([1.5, 2.5]),
        }
        decoded = serializer.loads(serializer.dumps(source))
        self.assertEqual(decoded["bootstrap"], source["bootstrap"])
        self.assertTrue(torch.equal(decoded["tensor"], source["tensor"]))

    def test_rpc_serializer_rejects_bad_header(self):
        serializer = DomainRpcSerializer(DomainPacketCodec())
        with self.assertRaises(Exception):
            serializer.loads(b"bad")

    def test_rejects_unknown_custom_tag_and_gapped_tensor(self):
        codec = DomainPacketCodec()
        unknown = {"schema": 1, "root": {"kind": "custom", "tag": "evil",
                   "fields": {"kind": "dict", "items": []}}, "tensors": []}
        with self.assertRaises(InvalidObjectError):
            codec.decode(json.dumps(unknown).encode(), b"")
        packet = codec.encode(torch.tensor([1], dtype=torch.int32))
        envelope = json.loads(packet.metadata)
        envelope["tensors"][0]["offset"] = 1
        with self.assertRaises(InvalidObjectError):
            codec.decode(json.dumps(envelope).encode(), b"\0" + b"".join(packet.tensor_bytes))

    def test_limits_are_enforced_before_decode_allocation(self):
        codec = DomainPacketCodec(limits=ProtocolLimits(max_metadata_bytes=16))
        with self.assertRaises(InvalidObjectError): codec.decode(b"x" * 17, b"")

    def test_state_flags_are_strict_and_consistent(self):
        with self.assertRaises(ValueError):
            SharedDomainListState.from_dict({"pending_shared_domains": 0,
                "checked_out_shared_batches": 0, "shared_idle": "yes",
                "failed": False, "failure_reason": None})
        with self.assertRaises(ValueError):
            SharedDomainListState.from_dict({"pending_shared_domains": 1,
                "checked_out_shared_batches": 0, "shared_idle": True,
                "failed": False, "failure_reason": None})

if __name__ == "__main__": unittest.main()
