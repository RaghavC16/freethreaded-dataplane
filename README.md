# nogil-data-plane

This package is the tensor-aware specialization of `nogil_rpc` used by
distributed alpha-beta-CROWN. It replaces the data-transfer and
shared-domain-list responsibilities of `RayBatchedDomainList`, not worker
scheduling or the verifier's ordinary `BatchedDomainList` implementation.

The deliberately small runtime consists of:

- `BatchedDomainListActor`: the shared state exposed as a `nogil_rpc` actor;
- `BatchedDomainListActorServer`: the driver's thin runtime/owner wrapper;
- `RpcBatchedDomainList`: the synchronous worker-side facade;
- `DomainRpcSerializer`: the adapter between RPC envelopes and domain packets;
- `DomainPacketCodec`: JSON object metadata and raw CPU tensor bytes.

Workers keep their existing local domain lists. A parent picked from the shared
actor must be paired with exactly one shared `add()` or, after its children are
committed locally and reported to the control plane,
`complete_pick_to_local()`.

Strict consumers use the opaque `checkout_id` returned by a successful pick.
`publish_children`, `complete_pick_to_local`, or `complete_pruned` terminates
that exact checkout; `donate` publishes local-parent children and is rejected
while shared work is checked out. Stale, foreign, and duplicate tokens fail
before mutation. Compatibility `add` retains its existing behavior.

`snapshot()` atomically returns job identity, a monotonic version, shared
pending/checkout counts, and sticky failure state. `shared_idle` describes only
the actor and cannot establish global termination. Activation queries provide
sorting, minimum-domain records, and detached indexed records. The inspected
storage ignores `rev_order`, so `rev_order=True` is explicitly unsupported.

`RpcInputDomainList` and `InputDomainListActorServer` apply the checkout
protocol to both input-list variants while preserving the nine-field pick
tuple, keyword `add`, sorting, top-k/index queries, `get_progess`, and the
corrected `get_progress` alias.

| Optional activation capability | Status |
|---|---|
| `drop_lAs` | unsupported |
| clipping objective initialization | unsupported |
| `update_unstable_mask` | unsupported |
| shared counts/failure metadata | supported by `snapshot()` |

Unsupported mutation methods are absent and fail before storage changes.

## Minimal use

```python
codec = register_alpha_beta_crown_adapters(DomainPacketCodec())
server = BatchedDomainListActorServer(
    initial_domain_list,
    job_id="job-1",
    bind_host="0.0.0.0",
    advertise_host="node-1",
    codec=codec,
    validation_profile="alpha_beta_crown",
)
endpoint = server.start()

worker_codec = register_alpha_beta_crown_adapters(DomainPacketCodec())
shared = RpcBatchedDomainList(
    endpoint,
    device="cuda:0",
    worker_id="rank-0",
    codec=worker_codec,
)
domains = shared.pick_out(64)
```

`ADD`, `PICK_OUT`, and ambiguous connection failures are never retried. A
failure makes the job non-safe; the control plane must cancel workers and
return error/unknown. Only the control plane can decide global completion by
combining shared state with every worker's phase and local-domain count.
Disconnecting while a shared batch is checked out fails the data actor. An
idle disconnect does not, because only the control plane knows whether that
worker still owns local domains; worker health remains a control-plane duty.

Run tests with a Python environment containing PyTorch:

```bash
python -m unittest discover -s tests -v
```

The implementation reuses `nogil_rpc` connection management, object refs,
remote errors, and its single-worker executor per actor. It does not add a
second mailbox or actor thread. End-to-end free-threaded support still requires
a free-threaded-compatible PyTorch build.
