"""Prepare reproducibility evidence and concise delivery notes; no PDF report."""
import csv,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.io import write_json,write_csv,sha256,environment


def main():
    out=ROOT/'outputs/course_submission';evidence=ROOT/'docs/evidence/course_submission';evidence.mkdir(parents=True,exist_ok=True)
    notes=out/'delivery_notes';notes.mkdir(exist_ok=True)
    for key in ('stereo','C_ablation'):
        dst=evidence/key;dst.mkdir(exist_ok=True)
        for p in (out/key).glob('*.csv'):shutil.copyfile(p,dst/p.name)
        shutil.copyfile(out/key/'summary.json',dst/'summary.json')
    for name in ('depth_pro_nyuv2','unidepth_v2_nyuv2','depth_pro_C_cuda','unidepth_v2_C_cuda'):
        src=ROOT/'docs/evidence/full_benchmark_20261029'/name
        if not src.exists() and name.startswith('unidepth'):continue
        dst=evidence/name;dst.mkdir(exist_ok=True)
        for p in src.glob('*.json'):shutil.copyfile(p,dst/p.name)
        for p in src.glob('*.csv'):shutil.copyfile(p,dst/p.name)
    for model in ('dav2','vda'):
        run=out/('rollercoaster_'+model);dst=evidence/('rollercoaster_'+model);dst.mkdir(exist_ok=True)
        for name in ('metadata.json','config_resolved.json','timing.json','manifest.json','per_frame_timing.csv'):
            shutil.copyfile(run/name,dst/name)
        env=json.loads((run/'environment.json').read_text());write_json(dst/'software_environment.json',{k:env[k] for k in ('python','packages')})
    for name in ('video_comparison.json',):shutil.copyfile(out/name,evidence/name)
    original=json.loads((evidence/'depth_pro_C_cuda/summary.json').read_text())['means']
    ablation=json.loads((out/'C_ablation/summary.json').read_text())['means']
    table=[]
    for method,scores in {**original,**ablation}.items():
        table.append(dict(method=method,samples=654,**{k:scores[k] for k in ('abs_rel','rmse','delta1','si_boundary_f1')}))
    write_csv(evidence/'NYU_C_comparison.csv',table)
    cases=json.loads((out/'difficult_cases/cases.json').read_text());write_json(evidence/'difficult_cases.json',cases)
    preview=ROOT/'docs/figures/course_delivery';preview.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(out/'rollercoaster_frame_023.jpg',preview/'rollercoaster.jpg')
    shutil.copyfile(out/'nyu_application/rgb_depth_application.png',preview/'nyu_application.png')
    for i in (1,2,3):shutil.copyfile(out/f'difficult_cases/case_{i}/comparison.png',preview/f'difficult_case_{i}.png')
    (notes/'README.md').write_text('''# Depth demo delivery (report deliberately excluded)

Start with `rollercoaster_comparison.mp4`: RGB, framewise DAV2-Small and offline
VDA-Small on exactly the same genuine original frames, FP16, original 24 FPS,
max external side 640 and network short-side setting 518 (network HxW 518x924).
Each method uses one whole-clip P2-P98 range and the same TURBO palette. Relative
colors are not a shared metric scale; finite zeros are valid relative endpoints.

`runs/` preserves float32 NPY, saved RGB, nonnegative finite support masks,
manifests, configs, model/source hashes, software versions, timing and movies.
`source/` contains the actual provided video. It has only 70 frames: read the
source-length note before submission. No frames were duplicated or interpolated
to manufacture 100. Timing is offline throughput; the first inference is not
excluded as warm-up. No streaming real-time claim is made.

`application/` contains a provided NYUv2 image, original/C metric predictions,
filled GT, background-blur output, a five-second recorded focus demonstration
and a point cloud. RAW/C effects share focus and blur settings. The point cloud
uses predicted focal length, assumed center/square pixels, and stride 4.

`difficult_cases/` contains three actual supplied NYUv2 test cases with RGB,
float RAW/C/GT, GT validity, full views, zooms, selection criteria and metrics.
`stereo/` adds all-finite-positive-GT MAE/coverage; non-occluded scores remain
separate. `C_ablation/` removes RGB or depth guidance on all 654 test images,
without tuning the fixed C parameters. `evidence/` contains reproducible numbers.

`self_recorded/` includes four optional 64-frame/5-FPS application comparisons
from the existing captures. These excerpts have no depth GT and are additional
demonstrations, not replacements for the official video. Original complete
captures and research predictions remain in the project dataset/output folders.

Verify every member against `checksums.sha256`. With numpy and opencv-python,
run `python replay_course_video.py --root .` to replay the exported arrays on CPU.
For model inference and all quantitative experiments, use the public project
repository and its scripts/dependencies; pretrained weights and large source
datasets are not duplicated inside this demo ZIP.

Sources: https://github.com/Xiaofeipao/monocular-depth-video-consistency
Video/DAV2: https://github.com/DepthAnything/Video-Depth-Anything and
https://github.com/DepthAnything/Depth-Anything-V2 . NYUv2: Silberman et al.,
ECCV 2012; https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html .
Metric baseline: https://github.com/apple/ml-depth-pro . Stereo: Middlebury v3.

The academic PDF report is intentionally not included at the user's request.
Canvas submission and acceptance of the 70-versus-100-frame exception remain
external steps. This archive does not assert that those steps are completed.
''',encoding='utf-8')
    (notes/'SOURCE_LENGTH_NOTE.md').write_text('''# Official source length exception

The brief requests the first 100 frames of davis_rollercoaster.mp4. The directly
linked official file was downloaded again and fully decoded: 70 frames, 24 FPS,
960x540, 1,809,560 bytes. It is byte-identical to the cloned official asset.
SHA256: 7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8

This delivery uses every genuine frame (0..69) at original FPS. Model comparison
RGB is resized identically to 640x360. The model's internal overlap/padding is
its official inference mechanism; saved outputs contain exactly 70 frames.
Ask the course staff to confirm that the full linked 70-frame asset is accepted,
or provide the intended longer version. This is not a claim of 100-frame completion.
''',encoding='utf-8')
    texts=[
        'Window/outdoor appearance versus filled depth: large GT-protocol metric error is concentrated near the windows. Filled GT and visible through-window structure can represent different surfaces; RGB alone does not establish physical truth. Local C refinement cannot resolve this large surface/scale disagreement.',
        'Office boundary difficulty: furniture/wall regions contain GT depth changes that RAW largely smooths away. C does not reconstruct the missing geometry; SI Boundary F1 remains near zero. This is not fixed by a small improvement in average pixel depth error.',
        'Refinement regression: bedroom furniture/bed boundaries receive a lower SI score after C, despite a very small change in AbsRel. Inspect the saved float arrays and zoom; the visible RAW/C difference can be subtle. Aggregate improvement does not mean every image improves.'
    ]
    lines=['# Three difficult cases from the provided NYUv2 test data','',
        'Cases are selected post hoc for analysis, not for parameter fitting. Scores use filled depths, crop [45:471,41:601], .001<GT<10m, prediction clipping and no scale alignment. Full-frame displays use a fixed 0.5-10m scale; error displays clip at 2m. Green edge overlays are illustrative log-jump 0.1 edges, not the official SI-F1 formula.','']
    for item,text in zip(cases,texts):
        a,b=item['metrics'];lines += [f'## Case {item["case"]}: {item["sample_id"]} ({item["scene"]})','',text,'',
            f'RAW -> C: AbsRel {a["abs_rel"]:.6f} -> {b["abs_rel"]:.6f}; RMSE {a["rmse"]:.6f} -> {b["rmse"]:.6f} m; SI F1 {a["si_boundary_f1"]:.6f} -> {b["si_boundary_f1"]:.6f}.','']
    (notes/'DIFFICULT_CASES.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copyfile(ROOT/'scripts/replay_course_video.py',notes/'replay_course_video.py')
    tests=out/'tests.log'
    if tests.exists():
        import re
        found=re.search(r'(\d+) passed',tests.read_text(encoding='utf-8',errors='replace'))
        if found:write_json(evidence/'tests.json',dict(passed=int(found[1]),command='python -m pytest tests -q',log_sha256=sha256(tests)))
    import torch
    previous=evidence/'execution_settings.json'
    if previous.exists():
        gpu=json.loads(previous.read_text())['video_GPU']
    elif torch.cuda.is_available():
        properties=torch.cuda.get_device_properties(0)
        gpu=dict(alias='GPU-A',CUDA_compute_capability=list(torch.cuda.get_device_capability()),CUDA_visible_memory_bytes=properties.total_memory)
    else:
        gpu=dict(alias='GPU-A',details='See saved inference metadata; export/replay does not require GPU')
    write_json(evidence/'execution_settings.json',dict(video_GPU=gpu,
        video=dict(precision='fp16',variant='Small/vits',input_size=518,network_hw=[518,924],saved_hw=[360,640],fps=24,warmup=0,inference='framewise DAV2; offline VDA',
        peak_memory='torch.cuda.max_memory_allocated, not reserved/system memory'),
        metric_baselines='Original benchmark execution settings/precision remain separately recorded; no cross-architecture speed ranking',
        software={k:environment()[k] for k in ('python','packages')},report_status='Paused by user; no academic report generated'))
    sources=['scripts/complete_submission_video.py','scripts/build_course_media.py','scripts/rescore_course_stereo.py','scripts/ablate_course_refinement.py',
        'scripts/prepare_course_delivery.py','scripts/replay_course_video.py','scripts/run_course_delivery.ps1','src/package_submission.py','src/exploration/edge_aware.py','src/io.py','src/data/output_schema.py','configs/submission.example.json']
    write_json(evidence/'source_checksums.json',{p:sha256(ROOT/p) for p in sources})
    print('Prepared evidence, annotated cases, source-length note and replay script',flush=True)


if __name__=='__main__':main()
