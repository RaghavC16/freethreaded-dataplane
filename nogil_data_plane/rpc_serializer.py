"""Tensor-aware serializer adapter for nogil_rpc."""

from __future__ import annotations

import struct
from typing import Any

from nogil_rpc.errors import SerializationError

from .domain_packet_codec import DomainPacketCodec


MAGIC = b"NDRP"
SCHEMA_VERSION = 1
HEADER = struct.Struct("!4sBQ")
HEADER_SIZE = HEADER.size


class DomainRpcSerializer:
    """Serialize RPC envelopes with JSON metadata and raw tensor buffers."""

    def __init__(self, codec: DomainPacketCodec | None = None) -> None:
        self.codec = codec or DomainPacketCodec()

    def dumps(self, value: Any) -> bytes:
        try:
            packet = self.codec.encode(value)
            raw = b"".join(packet.tensor_bytes)
            if len(raw) != packet.total_tensor_bytes:
                raise SerializationError("domain codec reported an invalid raw length")
            return HEADER.pack(MAGIC, SCHEMA_VERSION, len(packet.metadata)) + packet.metadata + raw
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(f"failed to encode domain RPC payload: {exc}") from exc

    def loads(self, payload: bytes) -> Any:
        try:
            if len(payload) < HEADER_SIZE:
                raise SerializationError("domain RPC payload is truncated")
            magic, version, metadata_length = HEADER.unpack(payload[:HEADER_SIZE])
            if magic != MAGIC:
                raise SerializationError("invalid domain RPC payload magic")
            if version != SCHEMA_VERSION:
                raise SerializationError(
                    f"domain RPC schema version {version} != {SCHEMA_VERSION}"
                )
            metadata_end = HEADER_SIZE + metadata_length
            if metadata_end > len(payload):
                raise SerializationError("domain RPC metadata length exceeds payload")
            return self.codec.decode(payload[HEADER_SIZE:metadata_end], payload[metadata_end:])
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(f"failed to decode domain RPC payload: {exc}") from exc
