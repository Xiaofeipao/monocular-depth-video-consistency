"""Export saved relative-video runs and scoring evidence with neutral names."""
from pathlib import Path,PureWindowsPath
import hashlib,json,shutil,zipfile
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def main():
    package=Path('outputs/relative_video_bundle')
    package.mkdir(parents=True,exist_ok=False)
    for old,new in [('dav2_tokyo','dav2_relative'),('vda_tokyo','vda_relative')]:
        target=package/'runs'/new
        shutil.copytree(Path('outputs/gpu_20261028')/old,target)
        for file in target.glob('*.json'):
            data=json.loads(file.read_text())
            def clean(value):
                if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
                if isinstance(value,list): return [clean(v) for v in value]
                if isinstance(value,str):
                    path=PureWindowsPath(value)
                    if path.is_absolute():
                        parts=list(path.parts[1:])
                        if parts and parts[0] in ('Video-Depth-Anything','Depth-Anything-V2'):
                            value='third_party/'+'/'.join(parts)
                        else:
                            value='local_asset/'+path.name
                    value=value.replace('\\','/')
                    return value.replace(old,new)
                return value
            data=clean(data)
            if file.name=='environment.json':
                data['python']='Python 3.x'
                data['platform']='desktop environment'
                data['packages']['torch']='CUDA-enabled PyTorch'
                data['packages']['torchvision']='compatible torchvision'
                data['public_export_note']='Machine identity and model runtime versions generalized; numeric observations unchanged'
            file.write_text(json.dumps(data,indent=2),encoding='utf-8')
    scoring=package/'scoring'
    scoring.mkdir()
    source=Path('outputs/tokyo_repro_20261029/scoring')
    for name in ('relative_comparison.json','settings.json','per_pair.csv'):
        shutil.copy2(source/name,scoring/name)
    report=json.loads((scoring/'relative_comparison.json').read_text())
    report['dataset']='relative_video_demo; 32 frames, no depth GT'
    (scoring/'relative_comparison.json').write_text(json.dumps(report,indent=2))
    (scoring/'common_masks').mkdir()
    for file in (source/'flow_masks').glob('*.npz'):
        with np.load(file) as arrays:
            np.savez_compressed(scoring/'common_masks'/file.name,common_mask=arrays['common_mask'])
    (package/'scripts').mkdir()
    shutil.copy2('scripts/compare_relative_video.py',package/'scripts/compare_relative_video.py')
    (package/'src').mkdir()
    for file in ('__init__.py','flow.py','io.py'):
        shutil.copy2(Path('src')/file,package/'src'/file)
    shutil.copy2('docs/evidence/gpu_20261028/upstream_source_identity.json',package/'upstream_source_identity.json')
    (package/'requirements-replay.txt').write_text('numpy==2.4.6\nopencv-python==5.0.0.93\n')
    (package/'README.md').write_text('''# Relative video reproducibility bundle

The neutral demo name is unrelated to any user's location. Upstream input is
Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4. Source revisions
are in upstream_source_identity.json and weight/input hashes in run metadata.
No checkpoint weights are bundled; obtain the exact hashed public Small weights.

Both complete 32-frame run directories contain floating-point NPY predictions,
saved RGB, valid PNGs, manifest, configuration, environment, timing and rendered
RGB/depth MP4. Numeric and media files are copied unchanged. Machine-specific
paths and environment identity are generalized in the public JSON export.
Metadata's historical null upstream_git_sha is supplemented by the source file.

## Replay scoring (CPU only)

python -m pip install -r requirements-replay.txt
python scripts/compare_relative_video.py --dav2 runs/dav2_relative --vda runs/vda_relative --output replay_scoring

The scoring implementation was extracted into this executable script during
export; it was run against the original predictions and reproduced both
historical aggregate scores exactly. This is not a claim that the historical
inline command had previously been preserved as a standalone file.

Expected DAV2 0.006369762085077744; VDA 0.0042277539539481365.
Reference spans 7.40632963180542 and 11.056855201721191. Frame 000000 is the
fixed reference. The 31-pair CSV, common masks and complete scoring settings are
included. Replay regenerates forward/backward flow, FB errors and common masks.
Run valid/ PNGs describe original prediction validity; the scoring common masks
explicitly allow zero relative values. No per-frame normalization or GT fitting.

## Regenerate model predictions (GPU)

From a checkout of the main project with the pinned upstream repositories:

python -m src.infer --input third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4 --output outputs/dav2_relative_repeat --model dav2 --repo third_party/Depth-Anything-V2 --checkpoint checkpoints/depth_anything_v2_vits.pth --device cuda --fp16 --encoder vits --input-size 518 --fps 10 --max-frames 32 --max-side 640 --vis-min 0 --vis-max 50 --warmup 0

python -m src.infer --input third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4 --output outputs/vda_relative_repeat --model vda --repo third_party/Video-Depth-Anything --checkpoint checkpoints/video_depth_anything_vits.pth --device cuda --fp16 --encoder vits --input-size 518 --fps 10 --max-frames 32 --max-side 640 --vis-min 0 --vis-max 50 --warmup 0

Historical cold timings are preserved, not a controlled cross-device speed test.
GPU inference can differ slightly by runtime/device; CPU replay of these saved
predictions is the direct route to reproduce the reported comparison.
All archive members are listed in FILES.sha256.
''',encoding='utf-8')
    (package/'FILES.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(package).as_posix()}\n' for p in sorted(package.rglob('*')) if p.is_file()))
    archive=Path('outputs/relative_video_demo.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in package.rglob('*'):
            if file.is_file(): z.write(file,Path('relative_video_demo')/file.relative_to(package))
    destination=Path('artifacts/relative_video_demo')
    destination.mkdir(parents=True,exist_ok=False)
    parts=[]
    with archive.open('rb') as stream:
        i=0
        while block:=stream.read(32*1024*1024):
            i+=1
            file=destination/f'relative_video_demo.zip.part{i:03d}'
            file.write_bytes(block)
            parts.append(dict(file=file.name,bytes=len(block),sha256=digest(file)))
    (destination/'archive_manifest.json').write_text(json.dumps(dict(archive='relative_video_demo.zip',bytes=archive.stat().st_size,sha256=digest(archive),parts=parts),indent=2))
    print('Bundle created',archive.stat().st_size,digest(archive),flush=True)


if __name__=='__main__': main()
