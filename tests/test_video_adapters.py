"""Adapter contracts using official transforms/stitching and synthetic models.

These tests do not load checkpoints or establish real-model prediction quality.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, MethodType

import numpy as np
import pytest
import torch

from src.models.depth_anything_v2_adapter import (
    DepthAnythingV2Adapter, _configure_cpu_attention, _import_official, _runtime, _transform,
    network_input_shape,
)
from src.models.video_depth_anything_adapter import VideoDepthAnythingAdapter, window_metadata


ROOT = Path(__file__).resolve().parents[1]


def _source_module(path):
    spec = importlib.util.spec_from_file_location("video_test_official_transform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("height,width", [(720, 1280), (1280, 720), (360, 1600), (1600, 360), (479, 641), (30, 30)])
def test_both_official_transforms_match_common_shapes_and_values(height, width):
    da = _source_module(ROOT / "Depth-Anything-V2/depth_anything_v2/util/transform.py")
    vda = _source_module(ROOT / "Video-Depth-Anything/video_depth_anything/util/transform.py")
    rgb = np.random.default_rng(4).integers(0, 256, (height, width, 3), dtype=np.uint8)
    left, right = _transform(da, rgb, 518), _transform(vda, rgb, 518)
    assert left.dtype == np.float32
    assert left.shape[1:] == network_input_shape(height, width)
    np.testing.assert_array_equal(left, right)


def test_runtime_is_explicit_and_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _runtime("cpu", "float32")[1] == torch.device("cpu")
    with pytest.raises(ValueError, match="CPU"):
        _runtime("cpu", "float16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _runtime("cuda:0", "float16")
    with pytest.raises(ValueError, match="precision"):
        _runtime("cpu", "fp32")


def test_import_rejects_another_source_copy(tmp_path, monkeypatch):
    wrong = ModuleType("video_adapter_provenance_test")
    wrong.__file__ = str(tmp_path / "other" / "module.py")
    monkeypatch.setitem(sys.modules, wrong.__name__, wrong)
    expected = tmp_path / "expected"
    expected.mkdir()
    with pytest.raises(RuntimeError, match="does not originate"):
        _import_official(expected, wrong.__name__, (wrong.__name__,))


def test_framewise_uses_requested_tensor_device_and_keeps_zero(monkeypatch):
    # Simulates a CUDA-capable host with an explicitly requested CPU model.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    adapter = DepthAnythingV2Adapter.__new__(DepthAnythingV2Adapter)
    adapter.torch, adapter.device, adapter.precision, adapter.input_size = torch, torch.device("cpu"), "float32", 14
    adapter.metadata = {}
    adapter.transforms = _source_module(ROOT / "Depth-Anything-V2/depth_anything_v2/util/transform.py")

    class FakeModel(torch.nn.Module):
        def forward(self, tensor):
            assert tensor.device == torch.device("cpu")
            return tensor.new_zeros((len(tensor), *tensor.shape[-2:]))

        def infer_image(self, *_args):
            raise AssertionError("The upstream automatic device helper must not be called")

    adapter.model = FakeModel()
    actual = adapter.infer(np.zeros((11, 19, 3), dtype=np.uint8))
    assert actual.shape == (11, 19) and actual.dtype == np.float32
    assert np.all(actual == 0)
    assert adapter.metadata["network_input_hw"] == list(network_input_shape(11, 19, 14))


@pytest.mark.parametrize("length", [1, 22, 31, 32, 33, 55])
def test_official_window_stitching_trim_and_frame_order_with_synthetic_forward(length):
    # Exercise the actual upstream infer_video_depth implementation without
    # constructing its neural network or loading any model checkpoint.
    module = _import_official(
        ROOT / "Video-Depth-Anything", "video_depth_anything.video_depth",
        ("video_depth_anything", "utils"),
    )
    model = module.VideoDepthAnything.__new__(module.VideoDepthAnything)
    torch.nn.Module.__init__(model)
    model.metric = False

    class FakeHead(torch.nn.Module):
        def forward(self, _features, _patch_h, _patch_w, _length):
            return None

    model.head = FakeHead()

    def synthetic_forward(self, frames):
        _, t, _, h, w = frames.shape
        self.head(None, h // 14, w // 14, t)
        # Recover RGB frame IDs after the official transform. All pixels in
        # each synthetic input carry the same ID; distinct frames stay distinct.
        return (frames[:, :, 0] * 0.229 + 0.485) * 255.0 + 1.0

    model.forward = MethodType(synthetic_forward, model)
    adapter = VideoDepthAnythingAdapter.__new__(VideoDepthAnythingAdapter)
    adapter.model, adapter.torch, adapter.device = model, torch, torch.device("cpu")
    adapter.precision, adapter.input_size, adapter.metadata = "float32", 14, {}
    frames = np.repeat(np.arange(length, dtype=np.uint8)[:, None, None, None], 9 * 16 * 3).reshape(length, 9, 16, 3)
    original = frames.copy()
    result = adapter.infer(frames, 10.0)
    np.testing.assert_array_equal(frames, original)
    assert result.shape == (length, 9, 16)
    np.testing.assert_allclose(result, np.broadcast_to(np.arange(1, length + 1)[:, None, None], result.shape), atol=3e-4, rtol=0)
    settings = adapter.metadata["windowing"]
    assert settings["window_count"] == (length + 21) // 22
    assert settings["trimmed_frame_count"] == settings["padded_frame_count"]
    assert settings["stitch_output_frames_before_trim"] - settings["trimmed_frame_count"] == length
    assert len(adapter.metadata["observed_window_input_shapes"]) == settings["window_count"]


def test_window_boundary_metadata_is_official_not_naive_32_frame_chunking():
    info = window_metadata(32)
    assert info["window_starts"] == [0, 22]
    assert info["window_count"] == 2
    assert info["padded_frame_count"] == 22
    assert info["interpolated_overlap_frames"] == 8
    assert info["alignment_reference_keyframes"] == [0, 12]


@pytest.mark.parametrize("repository,namespace", [
    ("Depth-Anything-V2", "depth_anything_v2"),
    ("Video-Depth-Anything", "video_depth_anything"),
])
def test_cpu_attention_uses_same_official_reference_weights_without_global_patch(repository, namespace, monkeypatch):
    module = _import_official(ROOT / repository, namespace + ".dinov2_layers.attention", (namespace,))
    first = module.MemEffAttention(12, num_heads=3, qkv_bias=True).eval()
    untouched = module.MemEffAttention(12, num_heads=3).eval()
    original_forward = module.MemEffAttention.forward
    weights = {name: value.clone() for name, value in first.state_dict().items()}
    sample = torch.randn(2, 7, 12)
    expected = module.Attention.forward(first, sample)

    def no_cpu_xformers(*_args, **_kwargs):
        raise AssertionError("xFormers must not be called by CPU reference attention")

    monkeypatch.setattr(module, "memory_efficient_attention", no_cpu_xformers, raising=False)
    monkeypatch.setattr(module, "XFORMERS_AVAILABLE", True)
    record = _configure_cpu_attention(torch.nn.Sequential(first), torch.device("cpu"))
    torch.testing.assert_close(first(sample), expected, rtol=0, atol=0)
    assert record["dino_reference_instance_count"] == 1
    assert record["dino_reference_instance_paths"] == ["0"]
    assert untouched.forward.__func__ is original_forward
    assert module.MemEffAttention.forward is original_forward
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_gpu_attention_instances_remain_official_without_needing_a_cuda_kernel():
    module = _import_official(ROOT / "Depth-Anything-V2", "depth_anything_v2.dinov2_layers.attention", ("depth_anything_v2",))
    attention = module.MemEffAttention(12, num_heads=3)
    original = attention.forward.__func__
    record = _configure_cpu_attention(torch.nn.Sequential(attention), torch.device("cuda:0"))
    assert attention.forward.__func__ is original
    assert record["backend"] == "official_upstream"
    assert record["dino_reference_instance_count"] == record["motion_xformers_disabled_instance_count"] == 0


def test_cpu_motion_attention_selects_official_bmm_and_leaves_unrelated_flags_alone(monkeypatch):
    module = _import_official(ROOT / "Video-Depth-Anything", "video_depth_anything.motion_module.motion_module", ("video_depth_anything", "utils"))
    temporal = module.TemporalAttention(query_dim=16, heads=2, dim_head=8, dropout=0.).eval()
    assert temporal._use_memory_efficient_attention_xformers is True
    unrelated = torch.nn.Identity()
    unrelated._use_memory_efficient_attention_xformers = True
    model = torch.nn.ModuleList([temporal, unrelated])
    sample = torch.randn(4, 3, 16)
    temporal._use_memory_efficient_attention_xformers = False
    expected = temporal(sample, video_length=4)
    temporal._use_memory_efficient_attention_xformers = True
    record = _configure_cpu_attention(model, "cpu")
    assert record["motion_xformers_disabled_instance_count"] == 1
    assert temporal._use_memory_efficient_attention_xformers is False
    assert unrelated._use_memory_efficient_attention_xformers is True
    torch.testing.assert_close(temporal(sample, video_length=4), expected, rtol=0, atol=0)
