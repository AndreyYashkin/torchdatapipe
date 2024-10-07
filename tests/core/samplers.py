import unittest
import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True

    from torchdatapipe.core.samplers import (
        ListSampler,
        ConcatSampler,
        ConcatDatasetSampler,
        StackDatasetSampler,
        DistributedSampler,
        PytorchLightningSampler,
    )
except ImportError:
    TORCH_AVAILABLE = False


class UnitTestSamplers(unittest.TestCase):
    def get_samplers(self):
        samplers = {
            "EmptySampler": ListSampler(list(), shuffle=True),
            "ListSampler": ListSampler(list(range(10)), shuffle=True),
            "ConcatSampler": ConcatSampler(
                [ListSampler(list(range(10))), ListSampler(list(range(10, 20)))], shuffle=True
            ),
            "ConcatDatasetSampler": ConcatDatasetSampler(
                [ListSampler(list(range(10))), ListSampler(list(range(10)))], shuffle=True
            ),
            "StackDatasetSampler": StackDatasetSampler(
                [
                    ListSampler(list(range(10)), shuffle=True),
                    ListSampler(list(range(20)), shuffle=True),
                    ListSampler(list(range(5)), shuffle=True),
                ]
            ),
        }
        return samplers

    @unittest.skipUnless(TORCH_AVAILABLE, "no torch")
    def test_reproducibility(self):
        seed = 42
        samplers = self.get_samplers()
        for name, sampler in samplers.items():
            with self.subTest(sampler=name):
                rng_1 = np.random.default_rng(seed)
                sampler.set_rng(rng_1)
                indices_1 = list(sampler)

                rng_2 = np.random.default_rng(seed)
                sampler.set_rng(rng_2)
                indices_2 = list(sampler)

                self.assertListEqual(indices_1, indices_2)
                print(indices_1)

    @unittest.skipUnless(TORCH_AVAILABLE, "no torch")
    def test_distributed_reproducibility(self):
        seed = 42
        sampler = ListSampler(list(range(10)), shuffle=True)

        rng_1 = np.random.default_rng(seed)
        sampler.set_rng(rng_1)
        indices_1 = list(sampler)

        rng_2 = np.random.default_rng(seed)
        d_sampler = DistributedSampler(sampler)
        d_sampler.set_rng(rng_2)
        indices_2 = list(d_sampler)

        self.assertListEqual(indices_1, indices_2)

    @unittest.skipUnless(TORCH_AVAILABLE, "no torch")
    def test_distributed(self):
        seed = 42

        rng_0 = np.random.default_rng(seed)
        rng_1 = np.random.default_rng(seed)
        d_sampler_0 = DistributedSampler(
            ListSampler(list(range(10)), shuffle=True), rank=0, num_replicas=2
        )
        d_sampler_1 = DistributedSampler(
            ListSampler(list(range(10)), shuffle=True), rank=1, num_replicas=2
        )
        d_sampler_0.set_rng(rng_0)
        d_sampler_1.set_rng(rng_1)
        indices_0 = list(d_sampler_0)
        indices_1 = list(d_sampler_1)

        self.assertEqual(len(indices_0), 5)
        self.assertEqual(len(indices_1), 5)
        self.assertSetEqual(set(indices_0).intersection(set(indices_1)), set())
        self.assertSetEqual(set(indices_0 + indices_1), set(range(10)))

    @unittest.skipUnless(TORCH_AVAILABLE, "no torch")
    def test_pytorch_lightning(self):
        seed = 42
        sampler = PytorchLightningSampler(ListSampler(list(range(10)), shuffle=True), seed=seed)

        sampler.set_epoch(0)
        indices_0 = list(sampler)

        sampler.set_epoch(1)
        # indices_1 =
        list(sampler)

        sampler.set_epoch(0)
        indices_2 = list(sampler)
        self.assertListEqual(indices_0, indices_2)
