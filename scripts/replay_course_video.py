"""Replay exported relative predictions using only numpy and opencv-python."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--output',default='replayed_comparison.mp4')
    a=p.parse_args();root=Path(a.root);out=root/a.output
    if out.exists():raise FileExistsError(out)
    runs=[root/'runs'/n for n in ('dav2_relative','vda_relative')]
    meta=[json.loads((r/'metadata.json').read_text()) for r in runs]
    records=[json.loads((r/'manifest.json').read_text()) for r in runs]
    if len(records[0])!=len(records[1]):raise ValueError('Frame mismatch')
    writer=None
    try:
        for pair in zip(*records):
            pieces=[]
            for j,(run,r) in enumerate(zip(runs,pair)):
                for kind,ext,key in [('rgb','png','rgb_sha256'),('depth','npy','depth_sha256')]:
                    if hashlib.sha256((run/kind/f'{r["id"]}.{ext}').read_bytes()).hexdigest()!=r[key]:raise ValueError('Saved file changed')
                rgb=cv2.imread(str(run/'rgb'/f'{r["id"]}.png'));z=np.load(run/'depth'/f'{r["id"]}.npy')
                lo,hi=meta[j]['visualization_range'];color=cv2.applyColorMap(np.uint8(np.clip((z-lo)/(hi-lo),0,1)*255),cv2.COLORMAP_TURBO)
                if j==0:pieces.append(rgb)
                pieces.append(color)
            image=np.hstack(pieces)
            if writer is None:
                writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),meta[0]['fps'],(image.shape[1],image.shape[0]))
                if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
            writer.write(image)
    finally:
        if writer:writer.release()
    print('Replayed',len(records[0]),'genuine frames; relative colors are not a shared metric scale')


if __name__=='__main__':main()
