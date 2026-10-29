"""TUM RGB-D ingestion: registered RGB/depth, timestamp association and poses.

Depth PNGs are preserved byte-for-byte and decoded with 1/5000 m scaling.
Poses are interpolated T_world_camera (qx,qy,qz,qw), never inverted silently.
"""
from bisect import bisect_left
from pathlib import Path, PurePosixPath
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import numpy as np

from src.io import sha256,write_json
from src.prepare_data import validate_manifest


TUM_SOURCES={
    'freiburg1_xyz': 'freiburg1',
    'freiburg1_desk': 'freiburg1',
    'freiburg3_sitting_static': 'freiburg3',
}
FORMAT_URL='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats'


def source_urls(sequence):
    family=TUM_SOURCES[sequence]
    filename=f'rgbd_dataset_{sequence}.tgz'
    return [f'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/{family}/{filename}',
            f'https://cvg.cit.tum.de/rgbd/dataset/{family}/{filename}']


def download_xyz_ranges(destination,workers=8):
    """Resumable verified byte-range acquisition of the immutable xyz archive."""
    from concurrent.futures import ThreadPoolExecutor
    from .http_range import HTTPRangeFile
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    url=source_urls('freiburg1_xyz')[0]
    size=448204271  # official Content-Length, ETag 1ab70def-4ae2a1dc2ae80
    remote=HTTPRangeFile(url,size,destination.parent/'tum_range_cache',block_size=1024*1024,byte_budget=1024**3)
    # Reuse earlier validated larger ranges rather than redownloading them.
    import re
    for cached in list(remote.cache.glob('*.bin')):
        try:
            start,end=map(int,cached.stem.split('-'))
            header=cached.with_suffix('.headers')
            ranges=re.findall(r'Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)',header.read_text(encoding='latin1'),re.I)
            if end-start+1<=remote.block_size or cached.stat().st_size!=end-start+1 or not ranges or tuple(map(int,ranges[-1]))!=(start,end,size):
                continue
            content=cached.read_bytes()
            for position in range(start,end+1,remote.block_size):
                stop=min(end,position+remote.block_size-1)
                child=remote.cache/f'{position}-{stop}.bin'
                child.write_bytes(content[position-start:stop-start+1])
                child.with_suffix('.headers').write_text(f'HTTP/1.1 206\nContent-Range: bytes {position}-{stop}/{size}\n',encoding='ascii')
        except (ValueError,OSError):
            continue
    count=(size+remote.block_size-1)//remote.block_size
    # Cached chunks survive interruption. Only validated complete ranges enter
    # the archive; no extractall, symlink or path traversal is used.
    def fetch(index):
        remote._block(index)
        return index  # Do not retain every archive chunk in Future results.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(fetch,range(count)),1):
            if i%8==0: print(f'Validated TUM chunks {i}/{count}',flush=True)
    partial=destination.with_suffix(destination.suffix+'.assembled')
    with partial.open('wb') as target:
        for i in range(count):
            target.write(remote._block(i))
    if partial.stat().st_size!=size: raise ValueError('Assembled archive size mismatch')
    with tarfile.open(partial,'r:gz') as archive:
        archive.getmember('rgbd_dataset_freiburg1_xyz/rgb.txt')
    partial.replace(destination)
    remote.close()
    receipt={'sequence':'freiburg1_xyz','url':url,'bytes':size,'sha256':sha256(destination),
             'acquisition':'verified resumable 1 MiB byte ranges','etag':'1ab70def-4ae2a1dc2ae80'}
    write_json(destination.with_suffix('.receipt.json'),receipt)
    return receipt


def download_sequence(sequence,destination,maximum_bytes=1024**3):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError('Archive already exists; use it with the prepare command')
    errors=[]
    for url in source_urls(sequence):
        partial=destination.with_suffix(destination.suffix+'.part')
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'AIAA3201-depth-project/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response,partial.open('wb') as target:
                size=int(response.headers.get('Content-Length',0))
                if size>maximum_bytes:
                    raise ValueError('Archive exceeds download byte limit')
                count=0
                while True:
                    block=response.read(1024*1024)
                    if not block:
                        break
                    if count==0 and block[:2]!=b'\x1f\x8b':
                        raise ValueError('Server did not return a gzip archive')
                    count+=len(block)
                    if count>maximum_bytes:
                        raise ValueError('Archive exceeds download byte limit')
                    target.write(block)
                    if count//(16*1024*1024)!=(count-len(block))//(16*1024*1024):
                        print(f'Downloaded {count//1024//1024} MiB',flush=True)
                if size and count!=size:
                    raise ValueError('Incomplete archive')
                resolved_url=response.url
            partial.replace(destination)
            receipt={'sequence':sequence,'url':url,'resolved_url':resolved_url,
                     'bytes':destination.stat().st_size,'sha256':sha256(destination)}
            write_json(destination.with_suffix('.receipt.json'),receipt)
            return receipt
        except (OSError,ValueError) as error:
            # Windows curl uses Schannel/system roots. Keep TLS verification;
            # never use --insecure to work around Python's certificate store.
            if sys.platform=='win32' and 'CERTIFICATE_VERIFY_FAILED' in str(error) and shutil.which('curl.exe'):
                try:
                    command=['curl.exe','--fail','--location','--retry','2','--connect-timeout','30',
                             '--max-time','900','--max-filesize',str(maximum_bytes),'--output',str(partial),
                             '--write-out','%{url_effective}',url]
                    print('Using Windows system certificate store for official download',flush=True)
                    completed=subprocess.run(command,check=True,stdout=subprocess.PIPE,text=True)
                    with partial.open('rb') as candidate:
                        if candidate.read(2)!=b'\x1f\x8b':
                            raise ValueError('Server did not return a gzip archive')
                    partial.replace(destination)
                    receipt={'sequence':sequence,'url':url,'resolved_url':completed.stdout.strip(),
                             'bytes':destination.stat().st_size,'sha256':sha256(destination),'tls':'Windows Schannel verification'}
                    write_json(destination.with_suffix('.receipt.json'),receipt)
                    return receipt
                except (OSError,ValueError,subprocess.CalledProcessError) as curl_error:
                    errors.append(f'{url}: {curl_error}')
            errors.append(f'{url}: {error}')
    raise RuntimeError('Official TUM mirrors failed:\n'+'\n'.join(errors))


def read_table(text,columns):
    rows=[]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts=line.split()
        if len(parts)!=columns:
            raise ValueError('Unexpected TUM timestamp/pose row')
        stamp=float(parts[0])
        if not np.isfinite(stamp):
            raise ValueError('Nonfinite timestamp')
        rows.append((stamp,parts[1:]))
    rows.sort(key=lambda row:row[0])
    if len({r[0] for r in rows})!=len(rows):
        raise ValueError('Duplicate timestamps')
    return rows


def associate(rgb,depth,max_difference=.02):
    """Closest-first, one-to-one timestamp pairs with explicit time tolerance."""
    if max_difference<=0:
        raise ValueError('Timestamp tolerance must be positive')
    times=[r[0] for r in depth]
    candidates=[]
    for i,(stamp,_) in enumerate(rgb):
        left=bisect_left(times,stamp-max_difference)
        for j in range(left,len(depth)):
            delta=abs(times[j]-stamp)
            if times[j]>stamp+max_difference:
                break
            if delta<=max_difference:
                candidates.append((delta,i,j))
    used_rgb,used_depth=set(),set()
    pairs=[]
    for _,i,j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            used_rgb.add(i)
            used_depth.add(j)
            pairs.append((rgb[i],depth[j]))
    return sorted(pairs,key=lambda pair:pair[0][0])


def interpolate_pose(poses,stamp,max_gap=.05):
    times=[row[0] for row in poses]
    j=bisect_left(times,stamp)
    if j<len(times) and abs(times[j]-stamp)<1e-9:
        pose=np.asarray(poses[j][1],float)
    else:
        if j==0 or j==len(times):
            return None
        ta,tb=times[j-1],times[j]
        if tb-ta>max_gap:
            return None
        a,b=np.asarray(poses[j-1][1],float),np.asarray(poses[j][1],float)
        if not np.isfinite(np.r_[a,b]).all() or min(np.linalg.norm(a[3:]),np.linalg.norm(b[3:]))<1e-8:
            raise ValueError('Invalid pose quaternion')
        fraction=(stamp-ta)/(tb-ta)
        qa,qb=a[3:]/np.linalg.norm(a[3:]),b[3:]/np.linalg.norm(b[3:])
        dot=float(np.dot(qa,qb))
        if dot<0:
            qb,dot=-qb,-dot
        if dot>.9995:
            q=qa+fraction*(qb-qa)
        else:
            angle=np.arccos(np.clip(dot,-1,1))
            q=(np.sin((1-fraction)*angle)*qa+np.sin(fraction*angle)*qb)/np.sin(angle)
        pose=np.r_[a[:3]+fraction*(b[:3]-a[:3]),q/np.linalg.norm(q)]
    if not np.isfinite(pose).all() or np.linalg.norm(pose[3:])<1e-8:
        raise ValueError('Invalid interpolated pose')
    pose[3:]/=np.linalg.norm(pose[3:])
    return pose.tolist()


def _member_bytes(archive,name):
    if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
        raise ValueError('Unsafe archive member')
    member=archive.getmember(name)
    if not member.isfile():
        raise ValueError('Archive member is not a regular file')
    if member.size>10*1024*1024:
        raise ValueError('Unexpectedly large sample member')
    with archive.extractfile(member) as source:
        return source.read()


def prepare_sequence(archive_path,output,sequence='freiburg1_xyz',max_frames=16,target_fps=10.,max_difference=.02,max_pose_gap=.05,start_seconds=0.):
    if sequence not in TUM_SOURCES or max_frames<2 or target_fps<=0 or not np.isfinite(start_seconds) or start_seconds<0:
        raise ValueError('Unsupported sequence or invalid sampling parameters')
    archive_path,output=Path(archive_path),Path(output)
    if output.exists():
        raise FileExistsError(output)
    prefix=f'rgbd_dataset_{sequence}'
    with tarfile.open(archive_path,'r:gz') as archive:
        text={name:_member_bytes(archive,f'{prefix}/{name}').decode('utf-8') for name in ('rgb.txt','depth.txt','groundtruth.txt')}
        rgb,depth=read_table(text['rgb.txt'],2),read_table(text['depth.txt'],2)
        poses=read_table(text['groundtruth.txt'],8)
        pairs=associate(rgb,depth,max_difference)
        selected=[]
        next_stamp=rgb[0][0]+start_seconds
        for color,depth_row in pairs:
            stamp=color[0]
            if stamp+1e-7<next_stamp:
                continue
            pose=interpolate_pose(poses,stamp,max_pose_gap)
            if pose is None:
                continue
            selected.append((color,depth_row,pose))
            next_stamp=stamp+1/target_fps
            if len(selected)>=max_frames:
                break
        if len(selected)<2:
            raise ValueError('Too few synchronized RGB/depth/pose frames')
        wanted={f'{prefix}/{row[1][0]}' for color,d,_ in selected for row in (color,d)}
        payload={}
        # PNG members are not timestamp-sorted in the official tar. Repeated
        # random gzip seeks would decompress the entire archive dozens of times.
        with tarfile.open(archive_path,'r|gz') as stream:
            for member in stream:
                if member.name not in wanted:
                    continue
                path=PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not member.isfile() or member.size>10*1024*1024:
                    raise ValueError('Unsafe or oversized sample member')
                with stream.extractfile(member) as source:
                    payload[member.name]=source.read()
                if len(payload)==len(wanted):
                    break
        if set(payload)!=wanted:
            raise ValueError('Missing selected TUM PNG members')
        output.mkdir(parents=True)
        for directory in ('rgb','depth','valid'):
            (output/directory).mkdir()
        samples=[]
        for i,(color,d,pose) in enumerate(selected):
            key=f'{i:06d}'
            for row,kind in ((color,'rgb'),(d,'depth')):
                data=payload[f'{prefix}/{row[1][0]}']
                (output/kind/f'{key}.png').write_bytes(data)
            # Keep source invalid values; mask invalid GT rather than inpainting.
            from src.io import load_depth,write_image
            depth_m=load_depth(output/'depth'/f'{key}.png',1/5000.)
            write_image(output/'valid'/f'{key}.png',(np.isfinite(depth_m)&(depth_m>0)).astype(np.uint8)*255)
            sample_id=f'{sequence}_{key}' if start_seconds==0 else f'{sequence}_t{color[0]:.6f}'
            samples.append({'id':sample_id,'rgb':f'rgb/{key}.png','depth':f'depth/{key}.png',
                'mask':f'valid/{key}.png','depth_scale':1/5000.,'timestamp':color[0],'depth_timestamp':d[0],
                'rgb_depth_difference_seconds':abs(color[0]-d[0]),'pose_world_camera':pose,
                'source_rgb':color[1][0],'source_depth':d[1][0],
                'rgb_sha256':sha256(output/'rgb'/f'{key}.png'),'depth_sha256':sha256(output/'depth'/f'{key}.png')})
        for name,data in text.items():
            (output/name).write_text(data,encoding='utf-8')
    # ROS default intrinsics are recommended by TUM for the registered PNGs.
    write_json(output/'intrinsics.json',{'fx':525.,'fy':525.,'cx':319.5,'cy':239.5})
    manifest={'dataset':f'TUM RGB-D {sequence} selected public smoke subset','role':'optional_rgbd_video',
        'not_instructor_subset':True,'not_mandatory_clip':True,'sequence':sequence,
        'source_url':source_urls(sequence)[0],'format_reference':FORMAT_URL,'archive_sha256':sha256(archive_path),
        'sampling':{'target_fps':target_fps,'start_seconds':start_seconds,'max_rgb_depth_difference_seconds':max_difference,'max_pose_gap_seconds':max_pose_gap,
                    'actual_timestamps':[s['timestamp'] for s in samples],'uniform_frame_rate':False},
        'pose_convention':'T_world_camera: tx ty tz qx qy qz qw; interpolated at RGB timestamp',
        'video':{'playback_fps':target_fps,'timestamp_policy':'Actual sample timestamps for evaluation; nominal FPS for visualization only'},
        'protocol':{'min_depth':.001,'max_depth':10.,'crop':None,'edge_threshold':.08,'edge_tolerance':2},
        'intrinsics':'intrinsics.json','depth_scale_correction':'Already applied by TUM; no further ds correction',
        'samples':samples}
    write_json(output/'manifest.json',manifest)
    write_json(output/'validation.json',validate_manifest(output/'manifest.json'))
    return manifest
