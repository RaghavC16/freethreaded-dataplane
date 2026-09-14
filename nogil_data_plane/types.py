"""Small public records shared with the control plane and workers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

@dataclass(frozen=True)
class SharedDomainListEndpoint:
    host: str
    port: int
    actor_id: str
    job_id: str
    max_frame_size: int

    def __post_init__(self) -> None:
        if not self.host or not self.actor_id or not self.job_id:
            raise ValueError("host, actor_id, and job_id must not be empty")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if type(self.max_frame_size) is not int or self.max_frame_size <= 0:
            raise ValueError("max_frame_size must be a positive integer")

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


DomainListEndpoint = SharedDomainListEndpoint

@dataclass(frozen=True)
class SharedDomainListState:
    pending_shared_domains: int
    checked_out_shared_batches: int
    shared_idle: bool
    failed: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SharedDomainListState":
        try:
            if type(value["shared_idle"]) is not bool or type(value["failed"]) is not bool:
                raise TypeError("state flags must be booleans")
            if type(value["pending_shared_domains"]) is not int or type(value["checked_out_shared_batches"]) is not int:
                raise TypeError("state counts must be integers")
            result = cls(
                value["pending_shared_domains"], value["checked_out_shared_batches"],
                value["shared_idle"], value["failed"],
                value.get("failure_reason"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid shared-domain-list state") from exc
        if result.pending_shared_domains < 0 or result.checked_out_shared_batches < 0:
            raise ValueError("domain counts cannot be negative")
        if result.failure_reason is not None and not isinstance(result.failure_reason, str):
            raise ValueError("failure_reason must be a string or null")
        expected_idle = (result.pending_shared_domains == 0
                         and result.checked_out_shared_batches == 0
                         and not result.failed)
        if result.shared_idle != expected_idle:
            raise ValueError("shared_idle is inconsistent with counts/failure")
        return result

@dataclass(frozen=True)
class SharedDomainListSnapshot:
    """Atomic shared-frontier observation; it is not global termination."""
    job_id: str
    version: int
    pending_shared_domains: int
    checked_out_shared_batches: int
    shared_idle: bool
    failed: bool
    failure_reason: str | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SharedDomainListSnapshot":
        state = SharedDomainListState.from_dict(value)
        job_id, version = value.get("job_id"), value.get("version")
        if not isinstance(job_id, str) or not job_id or type(version) is not int or version < 0:
            raise ValueError("invalid shared-domain-list snapshot identity")
        return cls(job_id, version, state.pending_shared_domains,
                   state.checked_out_shared_batches, state.shared_idle,
                   state.failed, state.failure_reason)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

@dataclass(frozen=True)
class DomainRecord:
    lower_bound: Any
    upper_bound: Any
    threshold: Any
    history: Any
    split_history: Any
    depth: Any

    @classmethod
    def from_value(cls, value: Any) -> "DomainRecord":
        if isinstance(value, dict):
            source = value
            get = source.__getitem__
        else:
            get = lambda name: getattr(value, name)
        try:
            return cls(*(get(name) for name in (
                "lower_bound", "upper_bound", "threshold", "history",
                "split_history", "depth")))
        except (KeyError, AttributeError) as exc:
            raise ValueError("invalid domain query record") from exc

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

@dataclass(frozen=True)
class InputDomainListSnapshot:
    shared: SharedDomainListSnapshot
    output_device: Any = None
    storage_depth: int | None = None
    use_alpha: bool | None = None
    sort_index: int | None = None
    sort_descending: bool | None = None
    use_split_idx: bool | None = None
    spec_size: Any = None
    volume: Any = None
    all_volume: Any = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InputDomainListSnapshot":
        shared = SharedDomainListSnapshot.from_dict(value)
        return cls(shared, *(value.get(name) for name in (
            "output_device", "storage_depth", "use_alpha", "sort_index",
            "sort_descending", "use_split_idx", "spec_size", "volume",
            "all_volume")))

@dataclass(frozen=True)
class PickOutResult:
    status: Literal["data", "empty"]
    domains: dict | None
    state: SharedDomainListState

@dataclass(frozen=True)
class WorkerDomainState:
    phase: Literal["requesting", "processing", "committing", "waiting", "terminal"]
    local_pending_domains: int
    epoch: int

@dataclass(frozen=True)
class ProtocolLimits:
    max_metadata_bytes: int = 16 * 1024 * 1024
    max_tensor_bytes: int = 8 * 1024 * 1024 * 1024
    max_tensor_count: int = 100_000
    max_tree_depth: int = 128
    max_container_elements: int = 2_000_000
    max_tensor_rank: int = 32
    max_tensor_elements: int = 2**40
    max_string_bytes: int = 4 * 1024 * 1024

    def __post_init__(self):
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("all protocol limits must be positive")
