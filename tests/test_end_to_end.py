from __future__ import annotations

import pickle
import threading
import unittest

import torch

from nogil_data_plane import (
    BatchedDomainListActorServer,
    DomainPacketCodec,
    InputDomainListActorServer,
    RpcBatchedDomainList,
    RpcInputDomainList,
)


class FakeDomainList:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def pick_out(self, batch, device):
        selected, self.values = self.values[:batch], self.values[batch:]
        return {"ids": torch.tensor(selected, dtype=torch.int64, device=device)}

    def add(self, bounds, domain_data, check_infeasibility):
        self.values.extend(bounds["ids"].tolist())
        return torch.tensor(min(self.values)) if self.values else None

    def to_bytes(self):
        return pickle.dumps(self)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.server = BatchedDomainListActorServer(
            FakeDomainList(range(8)),
            "job-e2e",
            codec=DomainPacketCodec(),
            request_timeout=5,
            max_frame_size=16 * 1024 * 1024,
        )
        self.endpoint = self.server.start()
        self.clients = []

    def client(self, name):
        client = RpcBatchedDomainList(
            self.endpoint,
            worker_id=name,
            codec=DomainPacketCodec(),
            request_timeout=5,
        )
        self.clients.append(client)
        return client

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.server.stop()

    def test_shared_pick_to_shared_add(self):
        client = self.client("w0")
        selected = client.pick_out(2)
        self.assertEqual(selected["ids"].tolist(), [0, 1])
        self.assertTrue(client.has_checked_out_batch)
        lower = client.add({"ids": torch.tensor([100, 101])}, {}, False)
        self.assertEqual(lower.item(), 2)
        self.assertFalse(client.has_checked_out_batch)
        self.assertEqual(len(client), 8)

    def test_shared_pick_to_worker_local_completion(self):
        client = self.client("w0")
        children = client.pick_out(1)["ids"] + 10
        local = children.clone()
        client.complete_pick_to_local()
        self.assertEqual(local.tolist(), [10])
        state = client.state()
        self.assertEqual(state.pending_shared_domains, 7)
        self.assertEqual(state.checked_out_shared_batches, 0)

    def test_local_parent_donates_to_shared_without_checkout(self):
        client = self.client("w0")
        client.add({"ids": torch.tensor([50, 51])}, {}, False)
        self.assertEqual(len(client), 10)

    def test_two_workers_receive_disjoint_batches(self):
        clients = [self.client("w0"), self.client("w1")]
        barrier = threading.Barrier(2)
        results = [None, None]

        def pick(index):
            barrier.wait()
            results[index] = clients[index].pick_out(3)["ids"].tolist()

        threads = [
            threading.Thread(target=pick, args=(index,))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(set(results[0]) & set(results[1])), 0)
        clients[0].complete_pick_to_local()
        clients[1].complete_pick_to_local()
        self.assertEqual(self.server.state().pending_shared_domains, 2)

    def test_idle_attached_close_does_not_destroy_actor(self):
        client = self.client("w0")
        client.close()
        self.assertFalse(self.server.state().failed)
        self.assertEqual(self.server.state().pending_shared_domains, 8)

    def test_close_with_checkout_fails_actor_state(self):
        client = self.client("w0")
        client.pick_out(1)
        client.close()
        self.assertTrue(self.server.state().failed)


class FakeInputDomainList:
    output_device = "cpu"
    storage_depth = 7
    use_alpha = True
    sort_index = 2
    sort_descending = True
    use_split_idx = False
    spec_size = 3
    volume = 4.0
    all_volume = 8.0

    def __init__(self):
        self.values = [1, 2]

    def __len__(self):
        return len(self.values)

    def pick_out_batch(self, batch, device):
        selected, self.values = self.values[:batch], self.values[batch:]
        value = torch.tensor(selected, device=device)
        return ({}, value, value, value + 1, value, value, None, value, value)

    def add(self, lower_bound, *args, **kwargs):
        self.values.extend(lower_bound.tolist())

    def sort(self):
        self.values.sort()

    def get_topk_indices(self, k=1, largest=False, return_margin=False):
        indices = torch.arange(k)
        return (indices, indices.float()) if return_margin else indices

    def get_progess(self):
        return 0.5

    def __getitem__(self, index):
        return self.values[index]


class InputEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.server = InputDomainListActorServer(
            FakeInputDomainList(), "input-e2e", request_timeout=5,
            max_frame_size=16 * 1024 * 1024)
        self.endpoint = self.server.start()
        self.client = RpcInputDomainList(
            self.endpoint, worker_id="input-worker", request_timeout=5)

    def tearDown(self):
        self.client.close()
        self.server.stop()

    def test_metadata_pick_publish_and_state(self):
        self.assertEqual(self.client.storage_depth, 7)
        self.assertTrue(self.client.use_alpha)
        self.assertEqual(self.client.sort_index, 2)
        self.assertTrue(self.client.sort_descending)
        self.assertFalse(self.client.use_split_idx)
        self.assertEqual(self.client.spec_size, 3)
        self.assertEqual(self.client.volume, 4.0)
        self.assertEqual(self.client.all_volume, 8.0)
        picked = self.client.pick_out_batch(1)
        self.assertEqual(len(picked), 9)
        self.client.add(torch.tensor([9]), *picked[2:6])
        state = self.server.state()
        self.assertEqual(state.shared.pending_shared_domains, 2)
        self.assertEqual(state.shared.checked_out_shared_batches, 0)

    def test_worker_failure_with_checkout_fails_actor(self):
        self.client.pick_out_batch(1)
        state = self.client.worker_failed("solver crashed")
        self.assertTrue(state.shared.failed)
        self.assertIn("solver crashed", state.shared.failure_reason)


if __name__ == "__main__":
    unittest.main()
