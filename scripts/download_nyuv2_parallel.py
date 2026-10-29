"""Resume verified range downloads of the assignment's canonical labeled MAT."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import subprocess
import shutil

URL='https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SIZE=2972037809
SHA='2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa'


def main():
    root=Path('data/nyuv2')
    root.mkdir(parents=True,exist_ok=True)
    final=root/'nyu_depth_v2_labeled.mat'
    if final.exists():
        existing=hashlib.sha256()
        with final.open('rb') as stream:
            for data in iter(lambda:stream.read(1048576),b''): existing.update(data)
        if existing.hexdigest()!=SHA: raise ValueError('Existing MAT differs from canonical SHA256')
        print('Existing canonical MAT verified; no download needed',flush=True)
        return
    curl=shutil.which('curl.exe') or shutil.which('curl')
    if curl is None: raise RuntimeError('curl is required')
    cache=Path('data/downloads/nyuv2_ranges')
    cache.mkdir(parents=True,exist_ok=True)
    block=32*1024*1024
    def fetch(start):
        end=min(SIZE,start+block)-1
        path=cache/f'{start}-{end}.bin'
        headers=path.with_suffix('.headers')
        if not (path.exists() and path.stat().st_size==end-start+1 and headers.exists() and f'bytes {start}-{end}/{SIZE}' in headers.read_text().lower()):
            subprocess.run([curl,'--fail','--silent','--show-error','--location','--retry','5',
                '--connect-timeout','20','--max-time','600','--range',f'{start}-{end}',
                '--dump-header',str(headers),'--output',str(path),URL],check=True)
            if path.stat().st_size!=end-start+1 or f'bytes {start}-{end}/{SIZE}' not in headers.read_text().lower():
                raise ValueError('Invalid Content-Range or response size')
        return start
    starts=list(range(0,SIZE,block))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,future in enumerate(as_completed([pool.submit(fetch,s) for s in starts]),1):
            future.result()
            print(f'Verified blocks {n}/{len(starts)}',flush=True)
    temporary=root/'nyu_depth_v2_labeled.mat.assembled'
    digest=hashlib.sha256()
    with temporary.open('wb') as destination:
        for start in starts:
            end=min(SIZE,start+block)-1
            with (cache/f'{start}-{end}.bin').open('rb') as source:
                for data in iter(lambda:source.read(1024*1024),b''):
                    digest.update(data)
                    destination.write(data)
    if digest.hexdigest()!=SHA: raise ValueError('Full MAT SHA256 mismatch')
    destination=root/'nyu_depth_v2_labeled.mat'
    if destination.exists(): raise FileExistsError(destination)
    temporary.rename(destination)
    print('Canonical labeled MAT downloaded; SHA256 verified',flush=True)


if __name__=='__main__': main()
