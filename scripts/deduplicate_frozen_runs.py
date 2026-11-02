"""Deduplicate byte-identical RGB/depth in completed RunWriter directories.

All file contents and paths remain available. Only completed immutable runs
(manifest + timing) qualify; no caches, datasets, model weights or PDFs change.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flows-only',action='store_true')
    args=parser.parse_args()
    seen={}; saved=0; pairs=[]
    paths=[]
    manifests=[] if args.flows_only else sorted((ROOT/'outputs').rglob('manifest.json'))
    for manifest in manifests:
        run=manifest.parent
        if not (run/'timing.json').is_file(): continue
        for kind,extension in [('rgb','png'),('depth','npy')]:
            paths.extend(sorted((run/kind).glob('*.'+extension)))
    if args.flows_only:
        for name in ('extended_video_validation_cpu','extended_video_validation_gpu518'):
            folder=ROOT/'outputs'/name
            if len(json.loads((folder/'summary.json').read_text(encoding='utf-8'))['clips'])!=6:
                raise ValueError('Only completed six-clip runs qualify')
            paths.extend(sorted(folder.glob('*/flow_farneback/*.npz')))
    for target in paths:
        target=target.resolve()
        if not target.is_relative_to(ROOT/'outputs'): raise ValueError('Outside outputs')
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        source=seen.get(digest)
        if source is None: seen[digest]=target; continue
        if os.path.samefile(source,target): continue
        if source.read_bytes()!=target.read_bytes(): raise ValueError('Hash collision')
        backup=target.with_suffix(target.suffix+'.dedup_backup')
        target.rename(backup)
        try: os.link(source,target)
        except BaseException:
            backup.rename(target)
            raise
        saved+=backup.stat().st_size; backup.unlink()
        pairs.append([source.relative_to(ROOT).as_posix(),target.relative_to(ROOT).as_posix()])
    path=ROOT/'outputs'/('frozen_flow_storage.json' if args.flows_only else 'frozen_run_storage.json')
    path.write_text(json.dumps(dict(bytes_saved=saved,pairs=pairs),indent=2),encoding='utf-8')
    print(json.dumps(dict(bytes_saved=saved,links=len(pairs))),flush=True)


if __name__=='__main__': main()
