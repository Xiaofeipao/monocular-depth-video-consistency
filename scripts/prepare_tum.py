"""Download one public TUM sequence, or prepare an already-downloaded archive."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.tum import TUM_SOURCES,download_sequence,download_xyz_ranges,prepare_sequence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=list(TUM_SOURCES),default='freiburg1_xyz')
    p.add_argument('--archive')
    p.add_argument('--download',action='store_true')
    p.add_argument('--range-download',action='store_true',help='Use verified resumable ranges for fr1/xyz')
    p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--fps',type=float,default=10)
    a=p.parse_args()
    archive=Path(a.archive or f'data/downloads/rgbd_dataset_{a.sequence}.tgz')
    if a.download and not archive.exists():
        if a.range_download:
            if a.sequence!='freiburg1_xyz': p.error('Range mode currently locks fr1/xyz source size only')
            download_xyz_ranges(archive)
        else:
            download_sequence(a.sequence,archive)
    manifest=prepare_sequence(archive,a.output,a.sequence,a.frames,a.fps)
    print(f"Prepared {len(manifest['samples'])} timestamp-associated RGB/depth/pose frames")


if __name__=='__main__':
    main()
