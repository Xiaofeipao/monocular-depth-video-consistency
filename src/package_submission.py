"""Package selected completed runs without source datasets or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='JSON {runs:[{name,path,role}],demo:path,notes:string}')
    p.add_argument('--output',default='depth_results.zip')
    a = p.parse_args(argv)
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = [r['name'] for r in manifest['runs']]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in ('.','..') for n in names):
        raise ValueError('Run names must be unique simple directory names')
    roles = {r['role'] for r in manifest['runs']}
    if not {'static_camera','moving_camera','phone_video'}.issubset(roles):
        raise ValueError('Include static_camera, moving_camera and phone_video runs')
    members = []
    allowed = {'.json','.csv','.npy','.npz','.png','.mp4','.ply','.txt','.md'}
    for row in manifest['runs']:
        run = manifest_path.parent/row['path']
        for filename in ('manifest.json','metadata.json','timing.json','visualization/rgb_depth.mp4'):
            if not (run/filename).is_file():
                raise ValueError(f'Incomplete run: {run/filename}')
        for file in sorted(run.rglob('*')):
            if file.is_file() and file.suffix.lower() in allowed:
                members.append((file,'depth_results/'+row['name']+'/'+file.relative_to(run).as_posix()))
    demo = manifest_path.parent/manifest['demo']
    if not (demo/'rgb_depth_application.png').is_file() or not (demo/'measurements.json').is_file():
        raise ValueError('Demo must contain application output and measurements')
    for file in sorted(demo.rglob('*')):
        if file.is_file() and file.suffix.lower() in allowed:
            members.append((file,'depth_results/demo/'+file.relative_to(demo).as_posix()))
    checksums = []
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for file,name in members:
            archive.write(file,name)
            h = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    h.update(block)
            checksums.append(h.hexdigest()+'  '+name)
        archive.writestr('depth_results/submission_manifest.json',json.dumps(manifest,indent=2))
        archive.writestr('depth_results/README.txt',manifest.get('notes','')+'\nDepth arrays: float32 NPY; consult each metadata.json for units.\nRun python -m src.demo --help for the application. See repository docs/implementation.md for reproduction.\n')
        archive.writestr('depth_results/checksums.sha256','\n'.join(checksums)+'\n')
    print(f'Wrote {a.output} ({len(members)} files)')


if __name__ == '__main__':
    main()
