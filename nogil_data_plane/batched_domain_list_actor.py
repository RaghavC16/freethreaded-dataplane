"""Shared BatchedDomainList state executed by one nogil_rpc actor executor."""

from __future__ import annotations

import pickle
from uuid import uuid4
from typing import Any

from nogil_rpc import remote

from .errors import ActorFailedError, DomainOperationError, InvalidStateError
from .types import DomainRecord, SharedDomainListState
from .validation import validate_alpha_beta_crown_add, validate_mapping_add


@remote
class BatchedDomainListActor:
    """Own the authoritative shared list and its worker checkout state."""

    def __init__(
        self,
        domains_or_bytes: Any,
        job_id: str,
        validation_profile: str = "mapping",
    ) -> None:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must not be empty")
        if validation_profile not in ("mapping", "alpha_beta_crown"):
            raise ValueError("unknown validation profile")
        if isinstance(domains_or_bytes, bytes):
            try:
                domains_or_bytes = pickle.loads(domains_or_bytes)
            except Exception as exc:
                raise ValueError("invalid BatchedDomainList bootstrap bytes") from exc
        self.job_id = job_id
        self._domains = domains_or_bytes
        self._validate_add = (
            validate_alpha_beta_crown_add
            if validation_profile == "alpha_beta_crown"
            else validate_mapping_add
        )
        self._registered_workers: set[str] = set()
        self._checkouts: dict[str, str] = {}
        self._failure_reason: str | None = None
        self._version = 0

    def _state(self) -> SharedDomainListState:
        pending = len(self._domains)
        checked_out = len(self._checkouts)
        failed = self._failure_reason is not None
        return SharedDomainListState(
            pending,
            checked_out,
            pending == 0 and checked_out == 0 and not failed,
            failed,
            self._failure_reason,
        )

    def _state_dict(self) -> dict[str, Any]:
        return self._state().to_dict()

    def _snapshot_dict(self) -> dict[str, Any]:
        return {**self._state_dict(), "job_id": self.job_id,
                "version": self._version}

    def _changed(self) -> None:
        self._version += 1

    def _require_worker(self, worker_id: str) -> None:
        if not isinstance(worker_id, str) or worker_id not in self._registered_workers:
            raise InvalidStateError("worker is not registered with this domain actor")

    def _require_available(self) -> None:
        if self._failure_reason is not None:
            raise ActorFailedError(self._failure_reason)

    def _record_failure(self, exc: BaseException) -> None:
        if self._failure_reason is None:
            self._failure_reason = f"{type(exc).__name__}: {str(exc)[:800]}"

    def register_worker(self, worker_id: str, job_id: str) -> dict[str, Any]:
        self._require_available()
        if not isinstance(worker_id, str) or not worker_id:
            raise InvalidStateError("worker_id must not be empty")
        if job_id != self.job_id:
            raise InvalidStateError("worker job_id does not match domain actor")
        if worker_id not in self._registered_workers:
            self._registered_workers.add(worker_id)
            self._changed()
        return {"status": "ok", "state": self._state_dict()}

    def pick_out(self, worker_id: str, batch_size: int) -> dict[str, Any]:
        self._require_available()
        self._require_worker(worker_id)
        if type(batch_size) is not int or batch_size <= 0:
            exc = InvalidStateError("batch_size must be a positive integer")
            self._record_failure(exc)
            raise exc
        if worker_id in self._checkouts:
            exc = InvalidStateError(
                "complete the active shared checkout before another pick_out"
            )
            self._record_failure(exc)
            raise exc
        try:
            pending = len(self._domains)
            if pending == 0:
                return {
                    "status": "empty",
                    "domains": None,
                    "state": self._state_dict(),
                }
            domains = self._domains.pick_out(min(batch_size, pending), "cpu")
        except Exception as exc:
            wrapped = DomainOperationError("BatchedDomainList.pick_out failed")
            self._record_failure(wrapped)
            raise wrapped from exc
        checkout = str(uuid4())
        self._checkouts[worker_id] = checkout
        self._changed()
        return {
            "status": "data",
            "domains": domains,
            "checkout": checkout,
            "state": self._state_dict(),
        }

    def add(
        self,
        worker_id: str,
        bounds: dict,
        domain_data: dict,
        check_infeasibility: bool,
    ) -> dict[str, Any]:
        self._require_available()
        self._require_worker(worker_id)
        if type(check_infeasibility) is not bool:
            exc = InvalidStateError("check_infeasibility must be a boolean")
            self._record_failure(exc)
            raise exc
        try:
            self._validate_add(bounds, domain_data)
            global_lb = self._domains.add(
                bounds, domain_data, check_infeasibility
            )
        except Exception as exc:
            wrapped = DomainOperationError("BatchedDomainList.add failed")
            self._record_failure(wrapped)
            raise wrapped from exc
        self._checkouts.pop(worker_id, None)
        self._changed()
        return {
            "status": "ok",
            "global_lb": global_lb,
            "state": self._state_dict(),
        }

    def _require_checkout(self, worker_id: str, checkout: str) -> None:
        self._require_worker(worker_id)
        if not isinstance(checkout, str) or self._checkouts.get(worker_id) != checkout:
            raise InvalidStateError("checkout is stale or does not belong to worker")

    def publish_children(self, worker_id: str, checkout: str, bounds: dict,
                         domain_data: dict, check_infeasibility: bool) -> dict[str, Any]:
        self._require_available()
        self._require_checkout(worker_id, checkout)
        if type(check_infeasibility) is not bool:
            raise InvalidStateError("check_infeasibility must be a boolean")
        try:
            self._validate_add(bounds, domain_data)
            global_lb = self._domains.add(bounds, domain_data, check_infeasibility)
        except Exception as exc:
            wrapped = DomainOperationError("BatchedDomainList.add failed")
            self._record_failure(wrapped)
            self._changed()
            raise wrapped from exc
        del self._checkouts[worker_id]
        self._changed()
        return {"status": "ok", "global_lb": global_lb,
                "state": self._state_dict()}

    def donate(self, worker_id: str, bounds: dict, domain_data: dict,
               check_infeasibility: bool) -> dict[str, Any]:
        self._require_available()
        self._require_worker(worker_id)
        if worker_id in self._checkouts:
            raise InvalidStateError("donation is forbidden during a shared checkout")
        return self.add(worker_id, bounds, domain_data, check_infeasibility)

    def complete_pick_to_local(self, worker_id: str,
                               checkout: str | None = None) -> dict[str, Any]:
        self._require_available()
        self._require_worker(worker_id)
        active = self._checkouts.get(worker_id)
        if active is None:
            exc = InvalidStateError("no shared pick_out is active")
            self._record_failure(exc)
            raise exc
        if checkout is not None and checkout != active:
            raise InvalidStateError("checkout is stale or does not belong to worker")
        del self._checkouts[worker_id]
        self._changed()
        return {"status": "ok", "state": self._state_dict()}

    def complete_pruned(self, worker_id: str, checkout: str) -> dict[str, Any]:
        return self.complete_pick_to_local(worker_id, checkout)

    def close_worker(self, worker_id: str) -> dict[str, Any]:
        self._require_worker(worker_id)
        if worker_id in self._checkouts:
            self._record_failure(
                InvalidStateError("worker closed with checked-out shared work")
            )
        self._registered_workers.discard(worker_id)
        self._changed()
        return {"status": "ok", "state": self._state_dict()}

    def worker_failed(self, worker_id: str, reason: str) -> dict[str, Any]:
        if worker_id in self._checkouts:
            self._record_failure(
                InvalidStateError(
                    f"worker {worker_id!r} failed with checked-out work: "
                    f"{str(reason)[:500]}"
                )
            )
        self._registered_workers.discard(worker_id)
        self._changed()
        return {"status": "ok", "state": self._state_dict()}

    def fail(self, reason: str) -> dict[str, Any]:
        if self._failure_reason is None:
            self._failure_reason = str(reason)[:1000]
            self._changed()
        return self._state_dict()

    def state(self) -> dict[str, Any]:
        return self._state_dict()

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_dict()

    def sort(self) -> dict[str, Any]:
        self._require_available()
        try:
            self._domains.sort()
        except Exception as exc:
            self._record_failure(exc)
            self._changed()
            raise DomainOperationError("domain-list sort failed") from exc
        self._changed()
        return {"status": "ok", "snapshot": self._snapshot_dict()}

    @staticmethod
    def _record(value: Any) -> dict[str, Any]:
        return DomainRecord.from_value(value).to_dict()

    def get_min_domain(self, num: int, rev_order: bool = False) -> dict[str, Any]:
        self._require_available()
        if type(num) is not int or num < 0 or type(rev_order) is not bool:
            raise InvalidStateError("num must be non-negative and rev_order boolean")
        if rev_order:
            raise InvalidStateError("rev_order=True is unsupported by BatchedDomainList")
        records = [self._record(x) for x in self._domains.get_min_domain(num, False)]
        return {"records": records, "version": self._version}

    def get_item(self, index: int, version: int | None = None) -> dict[str, Any]:
        self._require_available()
        if type(index) is not int:
            raise InvalidStateError("index must be an integer")
        if version is not None and version != self._version:
            raise InvalidStateError("snapshot version is stale")
        if index < 0:
            index += len(self._domains)
        if index < 0 or index >= len(self._domains):
            raise IndexError("domain index out of range")
        return {"record": self._record(self._domains[index]),
                "version": self._version}
