"""Package selected completed runs without source datasets or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package_course(manifest,base,output):
    """Validate current roles and disclose a genuinely short source clip."""
    import cv2
    import numpy as np
    from .io import load_run,sha256,read_frames,read_bgr
    roles={r['role']:r for r in manifest['runs']}
    if len(roles)!=len(manifest['runs']) or set(roles)!={'framewise_relative','video_relative'}:
        raise ValueError('Need one framewise_relative and one video_relative run')
    names=[r['name'] for r in manifest['runs']]
    if len(set(names))!=len(names) or any(not n or Path(n).name!=n or n in ('.','..') or '\\' in n for n in names):raise ValueError('Unsafe or duplicate run name')
    def resolve(value): return (base/value).resolve()
    source=resolve(manifest['source_video']);cap=cv2.VideoCapture(str(source));source_fps=cap.get(cv2.CAP_PROP_FPS);source_count=0
    while cap.read()[0]:source_count+=1
    cap.release()
    requested=int(manifest.get('requested_frames',100));expected=min(requested,source_count)
    if expected<2:raise ValueError('Source video has too few frames')
    if source_count<requested and not manifest.get('short_source_acknowledged'):
        raise ValueError('Short source requires explicit acknowledgement; do not pad frames')
    original,_,_=read_frames(source,max_frames=requested,max_side=manifest.get('max_res',640))
    loaded=[];files=[]
    for role in ('framewise_relative','video_relative'):
        row=roles[role];run=resolve(row['path']);meta,records=load_run(run)
        if meta['depth_type']!='relative' or meta.get('synthetic'):raise ValueError('Real relative-depth runs required')
        if meta.get('source_sha256')!=sha256(source):raise ValueError('Wrong source video')
        if len(records)!=expected or meta['frame_indices']!=list(range(expected)) or abs(meta['fps']-source_fps)>1e-6:
            raise ValueError('Original consecutive frames and original FPS required')
        for i,r in enumerate(records):
            for kind,ext,key in [('depth','npy','depth_sha256'),('rgb','png','rgb_sha256')]:
                if sha256(run/kind/f'{r["id"]}.{ext}')!=r[key]:raise ValueError('Frozen run changed')
            if not np.array_equal(read_bgr(run/'rgb'/f'{r["id"]}.png'),original[i]):raise ValueError('Stored RGB differs from the designated source')
            depth=np.load(run/'depth'/f'{r["id"]}.npy');valid=cv2.imread(str(run/'valid'/f'{r["id"]}.png'),0)>0
            if depth.dtype!=np.float32 or not np.isfinite(depth).all() or (depth<0).any():raise ValueError('Invalid relative output')
            if not np.array_equal(valid,np.isfinite(depth)&(depth>=0)):raise ValueError('Incorrect relative support mask')
        loaded.append((meta,records))
        for p in sorted(run.rglob('*')):
            if p.is_file():files.append((p,'depth_results/runs/'+row['name']+'/'+p.relative_to(run).as_posix()))
    if [r['rgb_sha256'] for r in loaded[0][1]]!=[r['rgb_sha256'] for r in loaded[1][1]]:raise ValueError('Methods use different RGB')
    comparison=resolve(manifest['comparison']);cap=cv2.VideoCapture(str(comparison));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while cap.read()[0]:count+=1
    cap.release()
    if count!=expected or abs(fps-source_fps)>1e-6:raise ValueError('Comparison incomplete or FPS changed')
    app=resolve(manifest['application'])
    if not (app/'rgb_depth_application.png').is_file() or not (app/'application.json').is_file():raise ValueError('Application missing')
    cases=resolve(manifest['difficult_cases']);case_list=json.loads((cases/'cases.json').read_text(encoding='utf-8'))
    if len(case_list)!=3:raise ValueError('Three difficult cases required')
    for item in case_list:
        for name in ('case.json','rgb.png','depths.npz','GT_valid.png','comparison.png','zoom.png'):
            if not (cases/f'case_{item["case"]}'/name).is_file():raise ValueError('Case missing '+name)
    for key in ('application','difficult_cases','stereo','C_ablation','self_recorded','evidence'):
        if not manifest.get(key):continue
        folder=resolve(manifest[key])
        for p in sorted(folder.rglob('*')):
            if p.is_file():files.append((p,'depth_results/'+key+'/'+p.relative_to(folder).as_posix()))
    files.extend([(comparison,'depth_results/rollercoaster_comparison.mp4'),(source,'depth_results/source/davis_rollercoaster.mp4')])
    for p in resolve(manifest.get('notes_directory',manifest['evidence'])).iterdir():
        if p.is_file() and p.suffix in ('.md','.py','.txt'):files.append((p,'depth_results/'+p.name))
    root=Path(__file__).resolve().parents[1]
    def portable(value):
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items() if k not in ('platform','device_name','gpu_name')}
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,str):
            for prefix in (str(root)+'\\',root.as_posix()+'/'):
                if value.startswith(prefix):value=value[len(prefix):]
            return value.replace('\\','/')
        return value
    checksums=[];names=set()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path,name in files:
            if name in names:raise ValueError('Duplicate archive entry '+name)
            names.add(name);content=path.read_bytes()
            if path.suffix=='.json':content=json.dumps(portable(json.loads(content)),indent=2,ensure_ascii=False).encode('utf-8')
            z.writestr(name,content);checksums.append(hashlib.sha256(content).hexdigest()+'  '+name)
        status=dict(frames=expected,requested_frames=requested,fps=source_fps,source_sha256=sha256(source),same_RGB_verified=True,
            source_length_exception=source_count<requested,course_confirmation_pending=source_count<requested,report_included=False)
        content=json.dumps(status,indent=2).encode();z.writestr('depth_results/package_validation.json',content)
        checksums.append(hashlib.sha256(content).hexdigest()+'  depth_results/package_validation.json')
        z.writestr('depth_results/checksums.sha256','\n'.join(checksums)+'\n')
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC validation failed')
        for line in z.read('depth_results/checksums.sha256').decode().splitlines():
            digest,name=line.split('  ',1)
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('ZIP member hash mismatch')
    receipt={**status,'archive':Path(output).name,'bytes':Path(output).stat().st_size,'sha256':sha256(output),'member_checksums_verified':len(checksums)}
    Path(str(output)+'.sha256').write_text(receipt['sha256']+'  '+Path(output).name+'\n',encoding='ascii')
    Path(str(output)+'.validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='JSON {runs:[{name,path,role}],demo:path,notes:string}')
    p.add_argument('--output',default='depth_results.zip')
    a = p.parse_args(argv)
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema')=='course_2027':
        return package_course(manifest,manifest_path.parent,Path(a.output))
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
