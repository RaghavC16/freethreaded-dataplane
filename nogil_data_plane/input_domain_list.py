"""RPC support for input-split domain lists."""
from __future__ import annotations
import pickle
import threading
from typing import Any
from uuid import uuid4
from nogil_rpc import RpcRuntime, connect, remote
from .domain_packet_codec import DomainPacketCodec
from .errors import ActorFailedError, InvalidStateError, ProtocolError
from .rpc_serializer import DomainRpcSerializer
from .types import InputDomainListSnapshot, SharedDomainListEndpoint

@remote
class InputDomainListActor:
    """Own input storage and serialize checkout and query operations."""
    def __init__(self, storage_bytes: bytes, job_id: str) -> None:
        if not isinstance(job_id, str) or not job_id: raise ValueError("job_id must not be empty")
        try: self._domains = pickle.loads(storage_bytes)
        except Exception as exc: raise ValueError("invalid input-domain bootstrap bytes") from exc
        self.job_id, self._workers, self._checkouts = job_id, set(), {}
        self._version, self._failure = 0, None

    def _snapshot(self):
        pending, checked = len(self._domains), len(self._checkouts)
        failed = self._failure is not None
        result = {"job_id": self.job_id, "version": self._version,
            "pending_shared_domains": pending, "checked_out_shared_batches": checked,
            "shared_idle": pending == 0 and checked == 0 and not failed,
            "failed": failed, "failure_reason": self._failure}
        for name in ("output_device", "storage_depth", "use_alpha", "sort_index",
                     "sort_descending", "use_split_idx", "spec_size", "volume", "all_volume"):
            if hasattr(self._domains, name): result[name] = getattr(self._domains, name)
        return result

    def _available(self):
        if self._failure is not None: raise ActorFailedError(self._failure)
    def _worker(self, worker_id):
        if worker_id not in self._workers: raise InvalidStateError("worker is not registered")
    def _checkout(self, worker_id, checkout):
        self._worker(worker_id)
        if self._checkouts.get(worker_id) != checkout: raise InvalidStateError("stale or foreign checkout")

    def register_worker(self, worker_id, job_id):
        self._available()
        if not isinstance(worker_id, str) or not worker_id or job_id != self.job_id: raise InvalidStateError("invalid worker/job identity")
        if worker_id not in self._workers: self._workers.add(worker_id); self._version += 1
        return {"status": "ok", "state": self._snapshot()}

    def pick_out_batch(self, worker_id, batch_size):
        self._available(); self._worker(worker_id)
        if type(batch_size) is not int or batch_size <= 0: raise InvalidStateError("batch_size must be positive")
        if worker_id in self._checkouts: raise InvalidStateError("checkout is active")
        if len(self._domains) == 0: return {"status": "empty", "data": None, "state": self._snapshot()}
        try: data = self._domains.pick_out_batch(min(batch_size, len(self._domains)), "cpu")
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"[:1000]; self._version += 1; raise
        checkout = str(uuid4()); self._checkouts[worker_id] = checkout; self._version += 1
        return {"status": "data", "data": data, "checkout": checkout, "state": self._snapshot()}

    def _add(self, args, kwargs):
        try: result = self._domains.add(*args, **kwargs)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"[:1000]; self._version += 1; raise
        self._version += 1; return result

    def publish(self, worker_id, checkout, args, kwargs):
        self._available(); self._checkout(worker_id, checkout); result = self._add(args, kwargs)
        del self._checkouts[worker_id]; self._version += 1
        return {"status": "ok", "result": result, "state": self._snapshot()}
    def donate(self, worker_id, args, kwargs):
        self._available(); self._worker(worker_id)
        if worker_id in self._checkouts: raise InvalidStateError("donation forbidden during checkout")
        return {"status": "ok", "result": self._add(args, kwargs), "state": self._snapshot()}
    def complete(self, worker_id, checkout):
        self._available(); self._checkout(worker_id, checkout); del self._checkouts[worker_id]; self._version += 1
        return {"status": "ok", "state": self._snapshot()}
    def query(self, name, args):
        self._available()
        if name not in {"get_topk_indices", "get_progess", "__getitem__"}: raise InvalidStateError("unsupported query")
        result = self._domains[args[0]] if name == "__getitem__" else getattr(self._domains, name)(*args)
        return {"result": result, "version": self._version}
    def sort(self): self._available(); self._domains.sort(); self._version += 1; return {"status": "ok"}
    def snapshot(self): return self._snapshot()
    def close_worker(self, worker_id):
        self._worker(worker_id)
        if worker_id in self._checkouts and self._failure is None: self._failure = "worker closed with checked-out input domains"
        self._workers.discard(worker_id); self._version += 1
        return {"status": "ok", "state": self._snapshot()}

class RpcInputDomainList:
    """Synchronous facade preserving the input storage APIs and pick tuple."""
    def __init__(self, endpoint, device="cpu", *, worker_id="worker", codec=None,
                 connect_timeout=30.0, request_timeout=300.0):
        self.endpoint, self.worker_id, self._device = endpoint, worker_id, device
        self._codec, self._timeout = codec or DomainPacketCodec(), request_timeout
        self._lock, self._checkout, self._closed = threading.Lock(), None, False
        self._process = connect(endpoint.address, serializer=DomainRpcSerializer(self._codec),
            timeout=connect_timeout, submission_timeout=request_timeout, max_frame_size=endpoint.max_frame_size)
        self._actor = self._process.attach_actor(endpoint.actor_id)
        response = self._actor.register_worker.remote(worker_id, endpoint.job_id).get(request_timeout)
        if not isinstance(response, dict) or response.get("status") != "ok": raise ProtocolError("malformed registration")
    def _call(self, name, *args):
        if self._closed: raise ActorFailedError("input facade is closed")
        return getattr(self._actor, name).remote(*args).get(self._timeout)
    def pick_out_batch(self, batch_size, device=None):
        with self._lock:
            if self._checkout is not None: raise InvalidStateError("checkout is active")
            response = self._call("pick_out_batch", self.worker_id, batch_size)
            if response.get("status") == "empty": return None
            if response.get("status") != "data" or not isinstance(response.get("checkout"), str): raise ProtocolError("malformed pick response")
            self._checkout = response["checkout"]; target = self._device if device is None else device
            return self._codec.map_tensors(response["data"], lambda x: x.to(target))
    @property
    def checkout_id(self): return self._checkout
    def add(self, *args, **kwargs):
        with self._lock:
            if self._checkout is None: response = self._call("donate", self.worker_id, args, kwargs)
            else: response = self._call("publish", self.worker_id, self._checkout, args, kwargs); self._checkout = None
            return response.get("result")
    def publish_children(self, checkout, *args, **kwargs):
        with self._lock:
            if checkout != self._checkout: raise InvalidStateError("checkout is stale")
            response = self._call("publish", self.worker_id, checkout, args, kwargs); self._checkout = None; return response.get("result")
    def donate(self, *args, **kwargs):
        with self._lock:
            if self._checkout is not None: raise InvalidStateError("checkout is active")
            return self._call("donate", self.worker_id, args, kwargs).get("result")
    def complete_pruned(self, checkout=None):
        with self._lock:
            token = self._checkout if checkout is None else checkout
            if token is None or token != self._checkout: raise InvalidStateError("checkout is stale")
            self._call("complete", self.worker_id, token); self._checkout = None
    complete_pick_to_local = complete_pruned
    def sort(self):
        with self._lock: self._call("sort")
    def get_topk_indices(self, k=1, largest=False, return_margin=False):
        with self._lock: return self._call("query", "get_topk_indices", (k, largest, return_margin))["result"]
    def get_progess(self):
        with self._lock: return self._call("query", "get_progess", ())["result"]
    get_progress = get_progess
    def __getitem__(self, index):
        with self._lock: return self._call("query", "__getitem__", (index,))["result"]
    def snapshot(self):
        with self._lock: return InputDomainListSnapshot.from_dict(self._call("snapshot"))
    def state(self): return self.snapshot()
    def __len__(self): return self.snapshot().shared.pending_shared_domains
    def close(self):
        with self._lock:
            if self._closed: return
            try: self._call("close_worker", self.worker_id)
            finally: self._closed = True; self._actor.close(); self._process.close()

class InputDomainListActorServer:
    def __init__(self, domains, job_id, *, bind_host="127.0.0.1", port=0, codec=None,
                 request_timeout=300.0, max_frame_size=512*1024*1024):
        self.domains, self.job_id, self.bind_host, self.port = domains, job_id, bind_host, port
        self.codec, self.timeout, self.max_frame_size = codec or DomainPacketCodec(), request_timeout, max_frame_size
        self.runtime = self.process = self.actor = None
    def start(self):
        self.runtime = RpcRuntime(self.bind_host, self.port, serializer=DomainRpcSerializer(self.codec), max_frame_size=self.max_frame_size); self.runtime.start()
        host, port = self.runtime.address
        self.process = connect(f"{host}:{port}", serializer=DomainRpcSerializer(self.codec), timeout=self.timeout, submission_timeout=self.timeout, max_frame_size=self.max_frame_size)
        payload = self.domains.to_bytes() if callable(getattr(self.domains, "to_bytes", None)) else pickle.dumps(self.domains)
        self.actor = self.process.InputDomainListActor.remote(payload, self.job_id)
        return SharedDomainListEndpoint(host, port, self.actor.actor_id, self.job_id, self.max_frame_size)
    def stop(self):
        if self.actor is not None: self.actor.close()
        if self.process is not None: self.process.close()
        if self.runtime is not None: self.runtime.stop()
