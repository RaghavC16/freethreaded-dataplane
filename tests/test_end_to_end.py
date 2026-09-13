from __future__ import annotations

import pickle
import threading
import unittest

import torch

from nogil_data_plane import (
    BatchedDomainListActorServer,
    DomainPacketCodec,
    RpcBatchedDomainList,
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


if __name__ == "__main__":
    unittest.main()
