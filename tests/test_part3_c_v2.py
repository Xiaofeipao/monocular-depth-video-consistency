"""C v2的真正收窄、纹理保护、独立开发协议及保存/核验门禁。"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np
import pytest
from src.part3_c_v2.refine import Parameters, refine, denoise
from pipelines.part3_c_v2 import common as v, direction_c as pipeline
from pipelines.part3 import common as c
from pipelines.part3_studies import common as study
from src.analysis.scale_boundary import profile_shape


def test_constant_depth_cannot_copy_texture_and_scale_equivariance():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (35, 57, 3), dtype=np.uint8)
    z = np.full(image.shape[:2], 2., np.float32)
    for method in ('C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support'):
        out = refine(z, image, method=method)
        np.testing.assert_allclose(out, z, atol=1e-6)
    depth = rng.uniform(1, 4, z.shape).astype(np.float32)
    a = refine(depth, image)
    b = refine(3*depth, image)
    np.testing.assert_allclose(b, 3*a, rtol=2e-5, atol=2e-6)


def test_actual_blurred_step_narrows_without_overshoot_or_shift():
    gt = np.full((81, 121), 2., np.float32); gt[:, 61:] = 4.
    rgb = np.zeros((*gt.shape, 3), np.uint8); rgb[:, 61:] = 255
    z = np.exp(cv2.GaussianBlur(np.log(gt), (15, 15), 2))
    p = Parameters(denoise_blend=.5, iterations=5)
    output, diagnostics = refine(z, rgb, p, return_diagnostics=True)
    a = profile_shape(z[40], 10, 1.05)
    b = profile_shape(output[40], 10, 1.05)
    assert a['status'] == b['status'] == 'measurable'
    assert b['width_10_90_px'] < a['width_10_90_px']
    assert abs(b['midpoint_index']-a['midpoint_index']) < .4
    assert output.min() >= z.min()-1e-6 and output.max() <= z.max()+1e-6
    assert abs(diagnostics['log_change']).max() <= p.max_log_change+1e-6
    assert diagnostics['gate'].max() > 0
    absent = np.full_like(rgb, 127)
    full = refine(z, absent, p)
    no_shock = refine(z, absent, p, method='C_v2_no_sharpen')
    np.testing.assert_array_equal(full, no_shock)


def test_log_affine_ramp_is_not_mistaken_for_a_sharp_step():
    x = np.linspace(0, .5, 101)
    z = np.tile(np.exp(x), (61, 1)).astype(np.float32)
    rgb = np.zeros((*z.shape, 3), np.uint8);rgb[:, 51:] = 255
    # Isolate the shock operator from the denoising stage.
    p = Parameters(denoise_blend=0, iterations=5)
    out = refine(z, rgb, p, smoothed=z)
    np.testing.assert_allclose(out[10:-10, 10:-10], z[10:-10, 10:-10], rtol=2e-6)


def test_invalid_inputs_and_iteration_parameters_rejected():
    z = np.ones((8, 9), np.float32);rgb = np.zeros((8, 9, 3), np.uint8)
    for value in (np.nan, np.inf, 0., -1.):
        bad = z.copy();bad[1, 1] = value
        with pytest.raises(ValueError):refine(bad, rgb)
    with pytest.raises(ValueError):Parameters(iterations=1.5)
    with pytest.raises(ValueError):Parameters(step=2)
    with pytest.raises(ValueError):refine(z, rgb.astype(float))


def test_scene_partition_is_shared_and_disjoint_not_random_images():
    cfg, _, _, _ = v.configuration(c.ROOT/'configs/part3_c_v2/direction_c.yaml', 'depth_pro')
    data = [SimpleNamespace(scene='scene'+str(i//3), test_position=i, matlab_index_1based=i+1)
            for i in range(30)]
    rows = v.partition(data, cfg)
    assert rows == v.partition(data, cfg)
    tune = {r['scene'] for r in rows if r['subset'] == 'tune'}
    val = {r['scene'] for r in rows if r['subset'] == 'validation'}
    assert len(tune) == 8 and len(val) == 2 and not tune & val


def test_selection_uses_one_common_support_and_never_replaces_with_raw():
    rows = []
    profiles = []
    for m, si, width in [('RAW', .8, 2.), ('a', .2, 3.), ('b', .3, 2.5)]:
        rows.append(dict(sample_id='one', method=m, si_boundary_f1=si, primary_f1=.5,
                         texture_spurious_fraction=.01))
        profiles.append(dict(sample_id='one', profile_id=0, method=m, scene='s',
                             status='measurable', width_10_90_px=width))
    selected = v.rank_candidates(rows, profiles, ['a', 'b'])
    assert selected['selected'] == 'b'  # RAW better, still show actual method.
    profiles[-1]['status'] = 'weak_contrast'
    with pytest.raises(ValueError, match='common measurable'):v.rank_candidates(rows, profiles, ['a', 'b'])


def test_summary_comparison_allows_small_float_changes_but_not_wrong_method():
    v.equivalent(dict(selected='a', means=[.123456789], status=True),
                 dict(selected='a', means=[.123456790], status=True))
    with pytest.raises(ValueError):v.equivalent(dict(selected='a'), dict(selected='b'))
    with pytest.raises(AssertionError):v.equivalent(dict(value=.1), dict(value=.2))
    with pytest.raises(ValueError):v.equivalent(dict(status=True), dict(status=1))


def test_old_B_and_C_source_fingerprints_unchanged():
    for name in ('direction_b/depth_pro/development_v1', 'direction_c/depth_pro/evaluation_v1'):
        path = c.ROOT/'outputs/part3'/name/'run_metadata.json'
        if path.exists():assert c.read_json(path)['source_sha256'] == c.sources()
    path = c.ROOT/'outputs/part3/direction_c/depth_pro/diagnostics_v1/run_metadata.json'
    if path.exists():assert c.read_json(path)['source_sha256'] == study.sources()
    assert not any('part3_c_v2' in name for name in study.sources())
    assert any('part3_c_v2' in name for name in v.sources())


def test_provenance_no_overwrite_subset_rejected_and_tampering_detected(tmp_path):
    cfg, _, _, _ = v.configuration(c.ROOT/'configs/part3_c_v2/direction_c.yaml', 'depth_pro')
    output = tmp_path/'subset'
    record = v.begin(output, 'c_v2_development', cfg, 'depth_pro', {})
    c.write_json(output/'test.json', {'truth': True})
    v.seal(output, record, False)
    assert not v.checked(output, full=False)['formal_complete']
    with pytest.raises(ValueError, match='Subset'):v.checked(output)
    with pytest.raises(FileExistsError):v.begin(output, 'c_v2_development', cfg, 'depth_pro', {})
    c.write_json(output/'test.json', {'truth': False})
    with pytest.raises(ValueError, match='artifact changed'):v.checked(output, full=False)


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """Synthetic split still exercises the full pipeline; never a real benchmark."""
    monkeypatch.setattr(c, 'CROP', np.s_[:, :])
    def valid(z):
        assert z.dtype == np.float32 and z.shape == (61, 101)
        assert np.isfinite(z).all() and (z > 0).all()
        return z
    monkeypatch.setattr(c, 'valid_depth', valid)
    gt = np.full((61, 101), 2., np.float32);gt[:, 51:] = 4
    rgb = np.zeros((*gt.shape, 3), np.uint8);rgb[:, 51:] = 255
    raw = np.exp(cv2.GaussianBlur(np.log(gt), (15, 15), 2)).astype(np.float32)
    class Dataset:
        def __init__(self, development=False):
            self.test_indices_1based = np.array([101, 102, 103, 104] if development else [1, 2])
            self.development = development
            labels = np.ones(gt.shape, np.uint8);labels[:, 51:] = 2
            self.file = {name: np.stack([labels.T]*104) for name in ('labels', 'instances')}
        def __len__(self):return len(self.test_indices_1based)
        def __enter__(self):return self
        def __exit__(self, *a):pass
        def get_test_sample(self, i):
            return SimpleNamespace(test_position=i, matlab_index_1based=int(self.test_indices_1based[i]),
                                   scene=('dev' if self.development else 'test')+str(i%2),
                                   rgb=rgb.copy(), depth_m=gt.copy())
    monkeypatch.setattr(c, 'dataset', lambda mc, development=False: Dataset(development))
    monkeypatch.setattr(c, 'resources', lambda *a: {})
    def score(z, gt, boundary=True):
        scores = dict(c.compute_depth_metrics(z, gt, min_depth=.001, max_depth=10.))
        if boundary:scores['si_boundary_f1'] = float(np.mean(abs(np.diff(z, axis=1))))
        return scores
    monkeypatch.setattr(c, 'score', score)
    cfg, base, mc, diagnostic = v.configuration(c.ROOT/'configs/part3_c_v2/direction_c.yaml', 'depth_pro')
    cfg['bootstrap_resamples'] = 10
    cfg['case_positions'] = []
    base['models']['depth_pro']['baseline'] = str(tmp_path/'native')
    source = Path(base['models']['depth_pro']['baseline']);(source/'predictions').mkdir(parents=True)
    for i in range(2):
        sample = Dataset().get_test_sample(i)
        np.savez_compressed(source/'predictions'/(c.stem(sample)+'.npz'), depth_m=raw)
    return cfg, base, mc, diagnostic, Dataset, raw, rgb, source


def test_full_synthetic_development_freeze_evaluation_replay(mini, tmp_path, monkeypatch):
    cfg, base, mc, diagnostic, Dataset, raw, rgb, source = mini
    prepared = tmp_path/'prepared'
    development_predictions = tmp_path/'development_arrays';development_predictions.mkdir()
    (development_predictions/'predictions').mkdir()
    rows = v.partition([Dataset(True).get_test_sample(i) for i in range(4)], cfg)
    for row in rows:
        path = development_predictions/'predictions'/(row['sample_id']+'.npz')
        np.savez_compressed(path, depth_m=raw)
        row.update(prediction_sha256=c.sha(path), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    record = v.begin(prepared, 'c_v2_prepared', cfg, 'depth_pro', {}, development_predictions=str(development_predictions))
    c.write_csv(prepared/'development_manifest.csv', rows);v.seal(prepared, record, True)
    dev = tmp_path/'development'
    pipeline.develop(dev, prepared, cfg, mc, diagnostic, 'depth_pro')
    assert pipeline.verify(dev, cfg, base, mc, diagnostic, 'depth_pro')['passed']
    frozen = tmp_path/'frozen';pipeline.freeze(frozen, dev, cfg, 'depth_pro')
    assert c.read_json(frozen/'frozen_parameters.json')['test_images_used_for_selection'] == 0
    ref = tmp_path/'ref';classics = tmp_path/'classics'
    base['output_root'] = str(tmp_path/'old_output')
    target_classics = Path(base['output_root'])/'depth_pro'/diagnostic['output_name']
    target_classics.mkdir(parents=True)
    oldrows = []
    for i in range(2):
        sample = Dataset().get_test_sample(i);key = c.stem(sample)
        for method in ('RAW', 'C', 'Gaussian_log', 'RGB_guided_log'):
            z = raw if method == 'RAW' else denoise(raw, rgb) if method == 'C' else pipeline.classical(method, raw, rgb, diagnostic['classical'])
            folder = ref if method == 'C' else target_classics
            path = folder/'predictions'/method/(key+'.npz');path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, depth_m=z)
            oldrows.append(dict(sample_id=key, method=method, source_prediction_sha256=c.sha(source/'predictions'/(key+'.npz')),
                postprocess_seconds=0., postprocess_backend='synthetic_cpu', peak_gpu_memory_bytes='', **c.score(z, sample.depth_m)))
    c.write_csv(target_classics/'per_image.csv', oldrows)
    monkeypatch.setattr(c, 'baseline_files', lambda *a: (source, {}, {}))
    monkeypatch.setattr(pipeline.original, 'reference', lambda *a: (ref, {}))
    real_checked = study.checked
    monkeypatch.setattr(study, 'checked', lambda path,*a,**kw: dict(model='depth_pro', config=diagnostic) if Path(path) == target_classics else real_checked(path,*a,**kw))
    real_v_checked = v.checked
    monkeypatch.setattr(v, 'checked', lambda path,*a,**kw: {} if Path(path) in (ref, target_classics) else real_v_checked(path,*a,**kw))
    out = tmp_path/'evaluation'
    pipeline.evaluate(out, frozen, cfg, base, mc, diagnostic, 'depth_pro', 2)
    assert pipeline.verify(out, cfg, base, mc, diagnostic, 'depth_pro')['passed']
    assert len(c.read_csv(out/'per_image.csv')) == 14
    assert not c.read_json(out/'run_metadata.json')['formal_complete']
    for method in ('C_v2', 'C_v2_no_sharpen', 'C_v2_no_RGB_support'):
        assert len(list((out/'predictions'/method).glob('*.npz'))) == 2
    # Re-sealing a forged array does not bypass numerical replay.
    path = out/'predictions/C_v2/test_0000_mat_0001.npz'
    np.savez_compressed(path, depth_m=np.full((61, 101), 5., np.float32))
    record = c.read_json(out/'run_metadata.json')
    record['artifacts'][str(path.relative_to(out))] = c.sha(path)
    c.write_json(out/'run_metadata.json', record)
    c.write_json(out/'status.json', dict(status='subset_complete', metadata_sha256=c.sha(out/'run_metadata.json')))
    with pytest.raises(AssertionError):pipeline.verify(out, cfg, base, mc, diagnostic, 'depth_pro')
