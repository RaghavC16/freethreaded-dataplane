from __future__ import annotations

import unittest

import torch

from nogil_data_plane import (
    BatchedDomainListActor,
    ActorFailedError,
    InvalidStateError,
)


class FakeDomainList:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def pick_out(self, batch, device):
        selected, self.values = self.values[:batch], self.values[batch:]
        return {"ids": torch.tensor(selected, device=device)}

    def add(self, bounds, domain_data, check_infeasibility):
        self.values.extend(bounds["ids"].tolist())
        return torch.tensor(min(self.values)) if self.values else None


class ActorTests(unittest.TestCase):
    def setUp(self):
        self.actor = BatchedDomainListActor(
            FakeDomainList([1, 2, 3]), "job"
        )
        self.actor.register_worker("w0", "job")
        self.actor.register_worker("w1", "job")

    def test_checkout_add_and_local_completion(self):
        first = self.actor.pick_out("w0", 1)
        self.assertEqual(first["domains"]["ids"].tolist(), [1])
        self.assertEqual(first["state"]["checked_out_shared_batches"], 1)
        result = self.actor.add(
            "w0",
            {"ids": torch.tensor([10, 11])},
            {},
            False,
        )
        self.assertEqual(result["state"]["checked_out_shared_batches"], 0)

        self.actor.pick_out("w1", 1)
        before = len(self.actor._domains.values)
        result = self.actor.complete_pick_to_local("w1")
        self.assertEqual(result["state"]["checked_out_shared_batches"], 0)
        self.assertEqual(len(self.actor._domains.values), before)

    def test_donation_without_checkout(self):
        result = self.actor.add(
            "w0", {"ids": torch.tensor([20])}, {}, False
        )
        self.assertEqual(result["state"]["pending_shared_domains"], 4)

    def test_second_pick_is_fatal(self):
        self.actor.pick_out("w0", 1)
        with self.assertRaises(InvalidStateError):
            self.actor.pick_out("w0", 1)
        self.assertTrue(self.actor.state()["failed"])
        with self.assertRaises(ActorFailedError):
            self.actor.pick_out("w1", 1)

    def test_close_with_checkout_fails(self):
        self.actor.pick_out("w0", 1)
        state = self.actor.close_worker("w0")["state"]
        self.assertTrue(state["failed"])

    def test_wrong_job_and_unregistered_worker_are_rejected(self):
        with self.assertRaises(InvalidStateError):
            self.actor.register_worker("other", "wrong-job")
        with self.assertRaises(InvalidStateError):
            self.actor.pick_out("other", 1)


if __name__ == "__main__":
    unittest.main()
