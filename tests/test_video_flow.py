"""Direction, canvas units and padding tests with mocked neural flow output."""

import numpy as np
import pytest
import torch

from src.video.flow import FlowEstimator


def test_explicit_cpu_farneback_identical_frames_are_zero():
    image = np.random.default_rng(9).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    model = FlowEstimator(backend="farneback", device="cpu", iterations=3)
    forward, backward = model.estimate(image, image)
    assert forward.shape == backward.shape == (48, 64, 2)
    # Farneback's border estimates are not mathematically exact zero.
    assert np.mean(np.abs(forward)) < 0.001
    np.testing.assert_allclose(forward, 0, atol=0.05)
    np.testing.assert_allclose(backward, 0, atol=0.05)


def test_farneback_is_never_a_silent_cuda_fallback():
    with pytest.raises(ValueError, match="explicit CPU"):
        FlowEstimator(backend="farneback", device="cuda:0")
    with pytest.raises(ValueError, match="checkpoint"):
        FlowEstimator(backend="farneback", device="cpu", checkpoint="ignored.pth")


def test_downsampled_flow_rescales_xy_components_and_preserves_direction(monkeypatch):
    model = FlowEstimator(backend="farneback", device="cpu", max_long_edge=4)
    first = np.zeros((7, 10, 3), dtype=np.uint8)
    second = np.full_like(first, 100)

    def fake(a, b):
        direction = 1 if a.mean() < b.mean() else -1
        flow = np.zeros((*a.shape[:2], 2), dtype=np.float32)
        flow[...] = [direction * 2, direction * 3]
        return flow

    monkeypatch.setattr(model, "_one", fake)
    forward, backward = model.estimate(first, second)
    # Rounded estimation canvas is 3x4: x and y need different exact factors.
    np.testing.assert_allclose(forward[..., 0], 2 * 10 / 4)
    np.testing.assert_allclose(forward[..., 1], 3 * 7 / 3, atol=1e-6)
    np.testing.assert_array_equal(backward, -forward)
    assert model.metadata["flow_estimation_shape"] == [3, 4]


def test_raft_padding_is_removed_before_returning_original_canvas():
    from torchvision.models.optical_flow import Raft_Small_Weights

    model = FlowEstimator.__new__(FlowEstimator)
    model.backend, model.torch, model.device = "raft_small", torch, torch.device("cpu")
    model.max_long_edge, model.iterations, model.metadata = None, 2, {}
    model.transform = Raft_Small_Weights.DEFAULT.transforms()
    calls = []

    class FakeRaft(torch.nn.Module):
        def forward(self, a, b, *, num_flow_updates):
            calls.append((tuple(a.shape), float(a.mean()), float(b.mean()), num_flow_updates))
            assert a.shape == b.shape == (1, 3, 128, 136)
            assert num_flow_updates == 2
            sign = 1 if a.mean() < b.mean() else -1
            flow = torch.zeros((1, 2, 128, 136))
            flow[:, 0], flow[:, 1] = 4 * sign, -2 * sign
            # Extremely wrong padded values must never leak into the output.
            flow[:, :, 35:] = 999
            flow[:, :, :, 129:] = 999
            return [flow]

    model.model = FakeRaft()
    forward, backward = model.estimate(np.zeros((35, 129, 3), np.uint8), np.full((35, 129, 3), 255, np.uint8))
    assert forward.shape == backward.shape == (35, 129, 2)
    np.testing.assert_array_equal(forward, np.broadcast_to([4, -2], forward.shape))
    np.testing.assert_array_equal(backward, -forward)
    assert calls[0][1] == -1 and calls[0][2] == 1
    assert model.metadata["padding_right_bottom"] == [7, 93]


def test_invalid_flow_fails_instead_of_becoming_a_good_score(monkeypatch):
    model = FlowEstimator(backend="farneback", device="cpu")
    monkeypatch.setattr(model, "_one", lambda a, b: np.full((*a.shape[:2], 2), np.nan, np.float32))
    with pytest.raises(ValueError, match="finite"):
        model.estimate(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8))
