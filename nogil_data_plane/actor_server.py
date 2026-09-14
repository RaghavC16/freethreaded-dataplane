"""Driver-side lifecycle wrapper for the shared domain-list RPC actor."""

from __future__ import annotations

import pickle
import threading
from typing import Any

from nogil_rpc import ActorHandle, RemoteProcess, RpcRuntime, connect

from .domain_packet_codec import DomainPacketCodec
from .rpc_serializer import DomainRpcSerializer
from .types import (SharedDomainListEndpoint, SharedDomainListSnapshot,
                    SharedDomainListState)


class BatchedDomainListActorServer:
    """Start one data RPC runtime and retain the actor's owning handle."""

    def __init__(
        self,
        domains: Any,
        job_id: str,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        advertise_host: str | None = None,
        codec: DomainPacketCodec | None = None,
        validation_profile: str = "mapping",
        request_timeout: float = 300.0,
        max_frame_size: int = 512 * 1024 * 1024,
    ) -> None:
        if not job_id:
            raise ValueError("job_id must not be empty")
        if type(max_frame_size) is not int or not 0 < max_frame_size < 2**32:
            raise ValueError("max_frame_size must be between 1 and 2**32 - 1")
        self._domains = domains
        self.job_id = job_id
        self.bind_host = bind_host
        self.port = port
        self.advertise_host = advertise_host or bind_host
        self.codec = codec or DomainPacketCodec()
        self.validation_profile = validation_profile
        self.request_timeout = request_timeout
        self.max_frame_size = max_frame_size
        self._runtime: RpcRuntime | None = None
        self._owner_process: RemoteProcess | None = None
        self._owner_actor: ActorHandle | None = None
        self._endpoint: SharedDomainListEndpoint | None = None
        self._lock = threading.Lock()

    def _bootstrap_bytes(self) -> bytes:
        to_bytes = getattr(self._domains, "to_bytes", None)
        return to_bytes() if callable(to_bytes) else pickle.dumps(self._domains)

    def start(self) -> SharedDomainListEndpoint:
        with self._lock:
            if self._runtime is not None:
                raise RuntimeError("domain actor server is already started")
            runtime = RpcRuntime(
                host=self.bind_host,
                port=self.port,
                serializer=DomainRpcSerializer(self.codec),
                max_frame_size=self.max_frame_size,
            )
            runtime.start()
            self._runtime = runtime

        owner_process: RemoteProcess | None = None
        owner_actor: ActorHandle | None = None
        try:
            bound_host, bound_port = runtime.address
            connect_host = (
                "127.0.0.1"
                if bound_host in ("0.0.0.0", "::")
                else bound_host
            )
            owner_process = connect(
                f"{connect_host}:{bound_port}",
                serializer=DomainRpcSerializer(self.codec),
                timeout=self.request_timeout,
                submission_timeout=self.request_timeout,
                max_frame_size=self.max_frame_size,
            )
            owner_actor = owner_process.BatchedDomainListActor.remote(
                self._bootstrap_bytes(),
                self.job_id,
                self.validation_profile,
            )
            endpoint = SharedDomainListEndpoint(
                self.advertise_host,
                bound_port,
                owner_actor.actor_id,
                self.job_id,
                self.max_frame_size,
            )
        except BaseException:
            if owner_actor is not None:
                owner_actor.close()
            if owner_process is not None:
                owner_process.close()
            runtime.stop()
            with self._lock:
                self._runtime = None
            raise

        with self._lock:
            self._owner_process = owner_process
            self._owner_actor = owner_actor
            self._endpoint = endpoint
        return endpoint

    def _actor(self) -> ActorHandle:
        with self._lock:
            actor = self._owner_actor
        if actor is None:
            raise RuntimeError("domain actor server is not started")
        return actor

    def state(self) -> SharedDomainListState:
        value = self._actor().state.remote().get(timeout=self.request_timeout)
        return SharedDomainListState.from_dict(value)

    def mark_failed(self, reason: str) -> SharedDomainListState:
        value = self._actor().fail.remote(reason).get(timeout=self.request_timeout)
        return SharedDomainListState.from_dict(value)

    def snapshot(self) -> SharedDomainListSnapshot:
        value = self._actor().snapshot.remote().get(timeout=self.request_timeout)
        return SharedDomainListSnapshot.from_dict(value)

    def worker_failed(self, worker_id: str, reason: str) -> SharedDomainListState:
        response = self._actor().worker_failed.remote(
            worker_id, reason
        ).get(timeout=self.request_timeout)
        return SharedDomainListState.from_dict(response["state"])

    def stop(self) -> None:
        with self._lock:
            actor = self._owner_actor
            process = self._owner_process
            runtime = self._runtime
            self._owner_actor = None
            self._owner_process = None
            self._runtime = None
            self._endpoint = None
        if actor is not None:
            try:
                actor.close()
            except Exception:
                pass
        if process is not None:
            process.close()
        if runtime is not None:
            runtime.stop()

    def __enter__(self) -> "BatchedDomainListActorServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


BatchedDomainListActorHost = BatchedDomainListActorServer
