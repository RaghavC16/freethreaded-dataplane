# Readiness runbook

| Gate | Command or evidence |
|---|---|
| CPU suite | `.venv/bin/python -m unittest discover -s tests -v` |
| RPC suites | run ordinary and free-threaded commands in `rpc_freethread/docs/readiness.md` |
| wheels | build both wheels and install into fresh environments outside the checkouts |
| CUDA | run packet and facade CPU-to-CUDA-to-CPU tests on one GPU |
| consumer parity | run opt-in fixtures against the pinned read-only verifier checkout |

Record passed, failed, or not-run status; commits, executables, versions, exit
codes, wheel SHA-256 values, import paths, GPU device, and GIL state. No command
here publishes packages or changes verifier pins.

## Evidence recorded 2026-09-13

| Gate | Result | Evidence |
|---|---|---|
| P1 regression | passed | RPC CPython and free-threaded: 47 tests each; data plane: 20 tests; all exit 0 |
| P1 wheels | passed | installed 0.3.0 wheels in `/tmp/dependency-readiness-env`; installed RPC smoke exit 0 |
| P2 deadlines/shutdown | passed for package tests | deterministic socket backpressure and unfinished-thread reporting |
| P3 ownership | passed for package tests | token, stale/duplicate rejection, failure, and compatibility coverage |
| P4 consumer parity | not run | real verifier storage fixtures remain opt-in |
| P5 CUDA packet transfer | passed | torch 2.13.0+cu130, RTX 4060 Laptop GPU, exact float16 CPU→CUDA→CPU |
| P6 artifacts | passed | RPC SHA-256 `4865a85cb69806e09572d318c6b73a19a7daf10326f625188b1782bb1b0907c3`; data plane SHA-256 `7c749c641036def4979de26b7d052bbe5064087e9b6cac1f59aae20c1134b9ee` |

Tested source bases: RPC `7b8c007e27ae46a8fb3c0b78665dcc79c8075bac`;
data plane `6382fddb557c4849317a28f298fd5b9bf274c00d`. Pin the
eventual commits containing these changes for verifier integration.
