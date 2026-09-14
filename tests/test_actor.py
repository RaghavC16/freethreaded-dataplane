from __future__ import annotations

import unittest

import torch

from nogil_data_plane import (
    BatchedDomainListActor,
    ActorFailedError,
    InvalidStateError,
    InputDomainListActor,
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

    def test_strict_checkout_rejects_stale_token_without_mutation(self):
        picked = self.actor.pick_out("w0", 1)
        before = list(self.actor._domains.values)
        with self.assertRaises(InvalidStateError):
            self.actor.publish_children(
                "w0", "stale", {"ids": torch.tensor([9])}, {}, False)
        self.assertEqual(self.actor._domains.values, before)
        self.actor.complete_pruned("w0", picked["checkout"])
        with self.assertRaises(InvalidStateError):
            self.actor.complete_pruned("w0", picked["checkout"])

    def test_snapshot_identity_version_and_queries(self):
        initial = self.actor.snapshot()
        picked = self.actor.pick_out("w0", 1)
        changed = self.actor.snapshot()
        self.assertEqual(changed["job_id"], "job")
        self.assertGreater(changed["version"], initial["version"])
        self.actor.complete_pruned("w0", picked["checkout"])


class FakeInputList:
    output_device = "cpu"
    storage_depth = 4
    use_alpha = False
    sort_index = None
    sort_descending = False
    use_split_idx = True
    def __init__(self): self.values = [1, 2]
    def __len__(self): return len(self.values)
    def pick_out_batch(self, batch, device):
        values, self.values = self.values[:batch], self.values[batch:]
        t = torch.tensor(values, device=device)
        return ({}, t, t, t + 1, t, t, None, t, t)
    def add(self, lb, dm_l, dm_u, alpha, cs, threshold, **kwargs):
        self.values.extend(lb.tolist())
    def sort(self): self.values.sort()
    def get_topk_indices(self, k=1, largest=False, return_margin=False):
        indices = torch.arange(k)
        return (indices, indices.float()) if return_margin else indices
    def get_progess(self): return 0.5
    def __getitem__(self, index): return self.values[index]


class InputActorTests(unittest.TestCase):
    def test_nine_field_pick_and_strict_completion(self):
        import pickle
        actor = InputDomainListActor(pickle.dumps(FakeInputList()), "input-job")
        actor.register_worker("w", "input-job")
        picked = actor.pick_out_batch("w", 1)
        self.assertEqual(len(picked["data"]), 9)
        snapshot = actor.snapshot()
        self.assertEqual(snapshot["storage_depth"], 4)
        with self.assertRaises(InvalidStateError):
            actor.complete("w", "wrong")
        actor.complete("w", picked["checkout"])


if __name__ == "__main__":
    unittest.main()
