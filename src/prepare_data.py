"""Validate portable manifests, or export NYUv2 MATLAB labeled test samples."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import read_bgr,load_depth,sha256,write_json,write_image
from .metrics import evaluation_mask


def validate_manifest(path):
    path = Path(path)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    samples = dataset['samples']
    ids = [x['id'] for x in samples]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Manifest needs nonempty, unique sample IDs")
    protocol = dataset['protocol']
    if not 0 < protocol['min_depth'] < protocol['max_depth']:
        raise ValueError("Invalid depth interval")
    records = []
    for row in samples:
        rgb = read_bgr(path.parent/row['rgb'])
        depth = load_depth(path.parent/row['depth'],row.get('depth_scale',1.))
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB/depth dimensions differ: {row['id']}")
        mask = load_depth(path.parent/row['mask'])>0 if row.get('mask') else None
        valid = evaluation_mask(depth,protocol['min_depth'],protocol['max_depth'],mask,protocol.get('crop'),protocol.get('depth_bounds','inclusive'))
        records.append({"id":row['id'],"rgb_sha256":sha256(path.parent/row['rgb']),
            "depth_sha256":sha256(path.parent/row['depth']),"valid_pixels":int(valid.sum())})
    return {"manifest_sha256":sha256(path),"count":len(samples),"samples":records}


def export_nyu(mat_path,split_path,output):
    # MATLAB v7.3 files store images as [N,3,W,H], depths as [N,W,H].
    # Do not guess a split; use the supplied official 1-based testNdxs.
    import h5py
    from scipy.io import loadmat
    indices = np.asarray(loadmat(split_path)['testNdxs']).reshape(-1).astype(int)
    if len(indices) != 654 or len(set(indices)) != 654 or indices.min() < 1:
        raise ValueError("Standard NYUv2 split must contain 654 unique 1-based testNdxs")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'rgb').mkdir()
    (output/'depth').mkdir()
    samples = []
    with h5py.File(mat_path,'r') as f:
        images,depths = f['images'],f['depths']
        if images.shape[1:] != (3,640,480) or depths.shape[1:] != (640,480):
            raise ValueError("Unexpected NYUv2 axis order; inspect the MAT file before converting")
        for number in indices:
            i = int(number)-1
            key = f"nyu_{number:04d}"
            rgb = np.asarray(images[i]).transpose(2,1,0)
            depth = np.asarray(depths[i]).T.astype(np.float32)
            write_image(output/'rgb'/f'{key}.png',rgb[...,::-1])
            np.save(output/'depth'/f'{key}.npy',depth)
            samples.append({"id":key,"rgb":f"rgb/{key}.png","depth":f"depth/{key}.npy","depth_scale":1.})
    write_json(output/'manifest.json',{"dataset":"NYUv2 standard 654 test split",
        "source_mat_sha256":sha256(mat_path),"split_sha256":sha256(split_path),
        "protocol":{"min_depth":.001,"max_depth":10.,"depth_bounds":"exclusive","clip_predictions":True,"crop":[45,471,41,601],
                    "edge_threshold":.08,"edge_tolerance":2},"samples":samples})
    write_json(output/'validation.json',validate_manifest(output/'manifest.json'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--manifest',required=True)
    nyu = commands.add_parser('nyu')
    nyu.add_argument('--mat',required=True)
    nyu.add_argument('--split',required=True)
    nyu.add_argument('--output',required=True)
    a = p.parse_args(argv)
    if a.command == 'validate':
        print(json.dumps(validate_manifest(a.manifest),indent=2))
    else:
        export_nyu(a.mat,a.split,a.output)


if __name__ == '__main__':
    main()
