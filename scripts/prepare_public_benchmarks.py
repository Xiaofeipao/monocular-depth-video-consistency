"""Prepare PUBLIC NYUv2/KITTI smoke samples, never an instructor-provided split."""
import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.http_range import HTTPRangeFile,curl_download
from src.io import sha256,write_json,write_image
from src.prepare_data import validate_manifest

NYU_URL='https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SPLIT_URL='https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
KITTI_URL='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip'


def nyu(output,count,cache):
    import h5py
    import numpy as np
    from scipy.io import loadmat
    split=Path(cache)/'nyu_splits.mat'
    if not split.exists(): curl_download(SPLIT_URL,split)
    indices=np.asarray(loadmat(split)['testNdxs']).reshape(-1).astype(int)[:count]
    samples=[]
    with HTTPRangeFile(NYU_URL,2972037809,cache) as remote:
        with h5py.File(remote,'r') as file:
            images,depths=file['images'],file['depths']
            if images.shape[1:]!=(3,640,480) or depths.shape[1:]!=(640,480):
                raise ValueError('Unexpected canonical NYUv2 HDF5 dimensions')
            output.mkdir(parents=True,exist_ok=False)
            (output/'rgb').mkdir(); (output/'depth').mkdir()
            for number in indices:
                key=f'nyu_{number:04d}'
                rgb=np.asarray(images[int(number)-1]).transpose(2,1,0)
                depth=np.asarray(depths[int(number)-1]).T.astype(np.float32)
                write_image(output/'rgb'/f'{key}.png',rgb[...,::-1])
                np.save(output/'depth'/f'{key}.npy',depth)
                samples.append({'id':key,'rgb':f'rgb/{key}.png','depth':f'depth/{key}.npy','depth_scale':1.,
                                'matlab_one_based_index':int(number),'rgb_sha256':sha256(output/'rgb'/f'{key}.png'),
                                'depth_sha256':sha256(output/'depth'/f'{key}.npy')})
    write_json(output/'manifest.json',{'dataset':'PUBLIC NYUv2 smoke subset from standard test split; NOT instructor-provided',
        'not_instructor_subset':True,'source_url':NYU_URL,'source_bytes':2972037809,'split_url':SPLIT_URL,'split_sha256':sha256(split),
        'sampling':'first N indices of official testNdxs, deterministic software check only',
        'protocol':{'min_depth':.001,'max_depth':10.,'depth_bounds':'exclusive','clip_predictions':True,
                    'crop':[45,471,41,601],'edge_threshold':.08,'edge_tolerance':2},'samples':samples})
    write_json(output/'validation.json',validate_manifest(output/'manifest.json'))


def kitti(output,count,cache):
    # Official depth-completion validation selection is already cropped. This
    # is not the Eigen split and is not the teacher's outdoor subset.
    samples=[]
    with HTTPRangeFile(KITTI_URL,2010655006,cache) as remote:
        with zipfile.ZipFile(remote) as archive:
            names=set(archive.namelist())
            gt=sorted(n for n in names if '/val_selection_cropped/groundtruth_depth/' in n and n.endswith('.png'))[:count]
            if len(gt)!=count: raise ValueError('Official KITTI validation selection not found')
            output.mkdir(parents=True,exist_ok=False)
            (output/'rgb').mkdir(); (output/'depth').mkdir(); (output/'intrinsics').mkdir()
            for i,depth_name in enumerate(gt):
                rgb_name=depth_name.replace('/groundtruth_depth/','/image/').replace('_groundtruth_depth_','_image_')
                if rgb_name not in names: raise ValueError(f'Unpaired KITTI RGB: {rgb_name}')
                key=f'kitti_public_{i:04d}'
                for source,kind in [(rgb_name,'rgb'),(depth_name,'depth')]:
                    member=archive.getinfo(source)
                    if member.file_size>10*1024*1024: raise ValueError('Oversized sample image')
                    (output/kind/f'{key}.png').write_bytes(archive.read(source))
                intrinsics_name=depth_name.replace('/groundtruth_depth/','/intrinsics/').replace('_groundtruth_depth_','_camera_').removesuffix('.png')+'.txt'
                if intrinsics_name in names:
                    (output/'intrinsics'/f'{key}.txt').write_bytes(archive.read(intrinsics_name))
                samples.append({'id':key,'rgb':f'rgb/{key}.png','depth':f'depth/{key}.png','depth_scale':1/256.,
                    'source_rgb':rgb_name,'source_depth':depth_name,'rgb_sha256':sha256(output/'rgb'/f'{key}.png'),
                    'depth_sha256':sha256(output/'depth'/f'{key}.png')})
    write_json(output/'manifest.json',{'dataset':'PUBLIC KITTI depth-completion validation selection smoke subset; NOT instructor/Eigen split',
        'not_instructor_subset':True,'source_url':KITTI_URL,'source_bytes':2010655006,
        'protocol':{'min_depth':.001,'max_depth':80.,'crop':None,'input_already_cropped':True,
                    'edge_threshold':.08,'edge_tolerance':2},'samples':samples})
    write_json(output/'validation.json',validate_manifest(output/'manifest.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['nyuv2','kitti'])
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--cache',default='data/downloads/range_cache')
    a=p.parse_args()
    if not 1<=a.count<=16: p.error('Public smoke subset count must be 1..16')
    (nyu if a.dataset=='nyuv2' else kitti)(Path(a.output),a.count,a.cache)
    print(f'Prepared {a.count} public {a.dataset} samples; not a course-subset completion claim')


if __name__=='__main__': main()
