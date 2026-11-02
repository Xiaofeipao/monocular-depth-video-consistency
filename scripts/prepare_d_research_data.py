"""Acquire four official TUM sequences; retain synchronized study subsets.

New full archives are temporary acquisitions, hashed in receipts then removed
only after the sampled original PNGs/tables and resized manifest are validated.
Existing project datasets and archives are never deleted.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.tum import download_sequence,prepare_sequence
from src.io import sha256,write_json
from src.prepare_data import validate_manifest
from scripts.run_extended_video_validation import resized_tum

SEQUENCES=['freiburg1_desk','freiburg1_rpy','freiburg3_sitting_static','freiburg3_walking_xyz']


def main():
    base=ROOT/'data/public_samples/d_research'; base.mkdir(parents=True,exist_ok=True)
    for sequence in SEQUENCES:
        original=base/(sequence+'_original'); output=base/sequence
        if (output/'manifest.json').exists():
            validate_manifest(output/'manifest.json')
            print(sequence+': cached validated subset',flush=True); continue
        cache=ROOT/'.project_cache/tmp/d_research'; cache.mkdir(parents=True,exist_ok=True)
        archive=cache/f'rgbd_dataset_{sequence}.tgz'
        if not (original/'manifest.json').exists():
            if not archive.exists(): download_sequence(sequence,archive)
            prepare_sequence(archive,original,sequence,max_frames=64,target_fps=5,start_seconds=0.)
        resized_tum(original,output)
        validate_manifest(output/'manifest.json')
        data=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
        if len(data['samples'])!=64: raise ValueError('Need full preregistered 64-frame excerpt')
        receipt=dict(sequence=sequence,source_url=data['source_url'],full_archive_sha256=data['archive_sha256'],
            retained_original_manifest_sha256=sha256(original/'manifest.json'),resized_manifest_sha256=sha256(output/'manifest.json'),
            retained_samples=64,storage='Original selected PNGs + timestamp/pose tables retained; full acquisition archive temporary')
        write_json(base/(sequence+'_receipt.json'),receipt)
        if archive.exists():
            if not archive.resolve().is_relative_to((ROOT/'.project_cache/tmp/d_research').resolve()): raise ValueError('Unsafe cache path')
            if sha256(archive)!=data['archive_sha256']: raise ValueError('Archive changed')
            archive.unlink()
        print(sequence+': prepared 64 frames; temporary archive released',flush=True)


if __name__=='__main__': main()
