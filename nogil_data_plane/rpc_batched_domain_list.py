"""Synchronous worker facade over a shared nogil_rpc domain-list actor."""

from __future__ import annotations

import threading
from typing import Any

from nogil_rpc import ActorHandle, RemoteProcess, connect
from nogil_rpc.errors import RemoteError, RpcError

from .domain_packet_codec import DomainPacketCodec
from .errors import (
    ActorFailedError,
    DataPlaneCancelledError,
    InvalidStateError,
    ProtocolError,
    RemoteDataPlaneError,
)
from .rpc_serializer import DomainRpcSerializer
from .types import (DomainRecord, SharedDomainListEndpoint,
                    SharedDomainListSnapshot, SharedDomainListState)


class RpcBatchedDomainList:
    """Present the shared RPC actor as the synchronous BDL API used by BaB."""

    def __init__(
        self,
        endpoint: SharedDomainListEndpoint,
        device: Any = "cpu",
        stop_event: threading.Event | None = None,
        *,
        worker_id: str = "worker",
        codec: DomainPacketCodec | None = None,
        connect_timeout: float = 30.0,
        request_timeout: float = 300.0,
        submission_timeout: float | None = None,
    ) -> None:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must not be empty")
        self.endpoint = endpoint
        self.worker_id = worker_id
        self._device = device
        self._stop_event = stop_event
        self._codec = codec or DomainPacketCodec()
        self._request_timeout = request_timeout
        self._lock = threading.Lock()
        self._has_checked_out_batch = False
        self._checkout: str | None = None
        self._closed = False
        self._broken = False
        process: RemoteProcess | None = None
        actor: ActorHandle | None = None
        try:
            process = connect(
                endpoint.address,
                serializer=DomainRpcSerializer(self._codec),
                timeout=connect_timeout,
                submission_timeout=(request_timeout if submission_timeout is None
                                    else submission_timeout),
                max_frame_size=endpoint.max_frame_size,
            )
            actor = process.attach_actor(endpoint.actor_id)
            response = actor.register_worker.remote(
                worker_id, endpoint.job_id
            ).get(timeout=request_timeout)
            if not isinstance(response, dict) or response.get("status") != "ok":
                raise ProtocolError("malformed worker registration response")
        except BaseException:
            if actor is not None:
                actor.close()
            if process is not None:
                process.close()
            raise
        self._process = process
        self._actor = actor

    def _check_available(self) -> None:
        if self._closed or self._broken:
            raise ActorFailedError("data-plane client is closed or broken")
        if self._stop_event is not None and self._stop_event.is_set():
            raise DataPlaneCancelledError("control-plane cancellation is set")

    def _call(self, method_name: str, *args: Any) -> Any:
        self._check_available()
        try:
            method = getattr(self._actor, method_name)
            return method.remote(*args).get(timeout=self._request_timeout)
        except RemoteError as exc:
            self._broken = True
            raise RemoteDataPlaneError(
                exc.error_type, str(exc), fatal=True
            ) from exc
        except RpcError as exc:
            self._broken = True
            raise ActorFailedError(f"domain RPC failed: {exc}") from exc
        except BaseException:
            self._broken = True
            raise

    @staticmethod
    def _require_response(response: Any, status: str) -> dict[str, Any]:
        if not isinstance(response, dict) or response.get("status") != status:
            raise ProtocolError("malformed domain actor response")
        return response

    def pick_out(self, batch: int, device: Any = None) -> dict | None:
        if type(batch) is not int or batch <= 0:
            raise ValueError("batch must be positive")
        with self._lock:
            if self._has_checked_out_batch:
                raise InvalidStateError(
                    "add or complete_pick_to_local is required before pick_out"
                )
            response = self._call("pick_out", self.worker_id, batch)
            if not isinstance(response, dict):
                self._broken = True
                raise ProtocolError("malformed pick_out response")
            if response.get("status") == "empty":
                return None
            self._require_response(response, "data")
            domains = response.get("domains")
            if not isinstance(domains, dict):
                self._broken = True
                raise ProtocolError("pick_out domains must be a dictionary")
            self._has_checked_out_batch = True
            checkout = response.get("checkout")
            if not isinstance(checkout, str) or not checkout:
                self._broken = True
                raise ProtocolError("pick_out checkout identity is missing")
            self._checkout = checkout
            target = self._device if device is None else device
            return self._codec.map_tensors(domains, lambda tensor: tensor.to(target))

    def add(
        self,
        bounds: dict,
        d: dict,
        check_infeasibility: bool,
    ) -> Any:
        if type(check_infeasibility) is not bool:
            raise TypeError("check_infeasibility must be bool")
        with self._lock:
            response = self._require_response(
                self._call(
                    "add",
                    self.worker_id,
                    bounds,
                    d,
                    check_infeasibility,
                ),
                "ok",
            )
            self._has_checked_out_batch = False
            self._checkout = None
            return self._codec.map_tensors(
                response.get("global_lb"),
                lambda tensor: tensor.to(self._device),
            )

    def complete_pick_to_local(self, checkout: str | None = None) -> None:
        with self._lock:
            if not self._has_checked_out_batch:
                raise InvalidStateError("no shared pick_out is active")
            token = self._checkout if checkout is None else checkout
            if token != self._checkout:
                raise InvalidStateError("checkout is stale or not active")
            self._require_response(
                self._call("complete_pick_to_local", self.worker_id,
                           token),
                "ok",
            )
            self._has_checked_out_batch = False
            self._checkout = None

    @property
    def checkout_id(self) -> str | None:
        with self._lock:
            return self._checkout

    def publish_children(self, checkout: str, bounds: dict, d: dict,
                         check_infeasibility: bool) -> Any:
        with self._lock:
            if checkout != self._checkout:
                raise InvalidStateError("checkout is stale or not active")
            response = self._require_response(self._call(
                "publish_children", self.worker_id, checkout, bounds, d,
                check_infeasibility), "ok")
            self._has_checked_out_batch = False
            self._checkout = None
            return self._codec.map_tensors(
                response.get("global_lb"), lambda tensor: tensor.to(self._device))

    def complete_pruned(self, checkout: str | None = None) -> None:
        with self._lock:
            token = self._checkout if checkout is None else checkout
            if token is None or token != self._checkout:
                raise InvalidStateError("checkout is stale or not active")
            self._require_response(
                self._call("complete_pruned", self.worker_id, token), "ok")
            self._has_checked_out_batch = False
            self._checkout = None

    def donate(self, bounds: dict, d: dict, check_infeasibility: bool) -> Any:
        with self._lock:
            if self._checkout is not None:
                raise InvalidStateError("donation is forbidden during a checkout")
            response = self._require_response(self._call(
                "donate", self.worker_id, bounds, d, check_infeasibility), "ok")
            return self._codec.map_tensors(
                response.get("global_lb"), lambda tensor: tensor.to(self._device))

    def state(self) -> SharedDomainListState:
        with self._lock:
            value = self._call("state")
            return SharedDomainListState.from_dict(value)

    def snapshot(self) -> SharedDomainListSnapshot:
        with self._lock:
            return SharedDomainListSnapshot.from_dict(self._call("snapshot"))

    def sort(self) -> None:
        with self._lock:
            self._require_response(self._call("sort"), "ok")

    def get_min_domain(self, num: int, rev_order: bool = False) -> list[DomainRecord]:
        with self._lock:
            response = self._call("get_min_domain", num, rev_order)
            if not isinstance(response, dict) or not isinstance(response.get("records"), list):
                raise ProtocolError("malformed domain query response")
            return [DomainRecord.from_value(x) for x in response["records"]]

    def get_item(self, index: int, *, version: int | None = None) -> DomainRecord:
        with self._lock:
            response = self._call("get_item", index, version)
            if not isinstance(response, dict) or "record" not in response:
                raise ProtocolError("malformed domain item response")
            return DomainRecord.from_value(response["record"])

    def __getitem__(self, index: int) -> DomainRecord:
        return self.get_item(index)

    def worker_failed(self, reason: str) -> SharedDomainListState:
        with self._lock:
            response = self._require_response(
                self._call("worker_failed", self.worker_id, reason),
                "ok",
            )
            return SharedDomainListState.from_dict(response["state"])

    def __len__(self) -> int:
        return self.state().pending_shared_domains

    @property
    def has_checked_out_batch(self) -> bool:
        with self._lock:
            return self._has_checked_out_batch

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._broken:
                try:
                    self._call("close_worker", self.worker_id)
                except BaseException:
                    pass
            self._closed = True
            self._actor.close()
            self._process.close()

    def __enter__(self) -> "RpcBatchedDomainList":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RpcBatchedDomainList(num_domains={len(self)})"


TcpBatchedDomainList = RpcBatchedDomainList
