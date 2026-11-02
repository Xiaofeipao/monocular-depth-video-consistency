"""Read the public iBims-1 repository listing and download only core MAT data.

All files remain inside the project. The public share token is not a credential.
"""
import argparse
import base64
import json
from pathlib import Path
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import hashlib

ROOT=Path(__file__).resolve().parents[1]
HOST='https://dataserv.ub.tum.de'
TOKEN='m1455541'


def request(url,method='GET',depth=None):
    headers={'Authorization':'Basic '+base64.b64encode((TOKEN+':').encode()).decode(),'User-Agent':'Research dataset reader'}
    if depth is not None: headers['Depth']=str(depth)
    return urllib.request.urlopen(urllib.request.Request(url,method=method,headers=headers),timeout=45)


def listing(url):
    with request(url,'PROPFIND',1) as f: payload=f.read()
    xml=ET.fromstring(payload)
    rows=[]
    for r in xml.findall('{DAV:}response'):
        href=r.findtext('{DAV:}href')
        prop=r.find('.//{DAV:}prop')
        rows.append(dict(href=href,name=urllib.parse.unquote(href.rstrip('/').split('/')[-1]),bytes=int(prop.findtext('{DAV:}getcontentlength') or 0),directory=prop.find('.//{DAV:}collection') is not None))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--processed-mirror',action='store_true',help='Author-released MoGe 100-image mirror; not the original MAT archive')
    a=p.parse_args()
    out=ROOT/'data/downloads'; out.mkdir(parents=True,exist_ok=True)
    if a.processed_mirror:
        api='https://huggingface.co/api/datasets/Ruicheng/monocular-geometry-evaluation/tree/main'
        with urllib.request.urlopen(api,timeout=45) as f: files=json.load(f)
        entries=[r for r in files if r.get('path')=='iBims-1.zip']
        if len(entries)!=1: raise ValueError('Expected exactly iBims-1.zip in mirror')
        entry=entries[0]; size=entry['size']
        if not 0<size<256*1024**2: raise ValueError('Unexpected mirror size')
        url='https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/iBims-1.zip'
        target=out/'ibims1_moge_processed.zip'
        if not target.exists():
            with urllib.request.urlopen(url,timeout=90) as source,target.with_suffix('.zip.part').open('wb') as dst:
                count=0
                while True:
                    block=source.read(1024*1024)
                    if not block: break
                    dst.write(block); count+=len(block)
                    if count>size: raise ValueError('Unexpected payload size')
            if count!=size: raise ValueError('Incomplete ZIP')
            target.with_suffix('.zip.part').replace(target)
        if target.stat().st_size!=size: raise ValueError('Mirror size mismatch')
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        expected=entry.get('lfs',{}).get('oid')
        if expected and digest!=expected: raise ValueError('Mirror SHA256 mismatch')
        (out/'ibims_mirror_receipt.json').write_text(json.dumps(dict(url=url,bytes=size,sha256=digest,hub_entry=entry,processed_mirror=True,not_original_MAT=True),indent=2),encoding='utf-8')
        print('Downloaded processed iBims-1 mirror:',size,digest,flush=True)
        return
    url=HOST+'/public.php/dav/files/'+TOKEN+'/'
    try: rows=listing(url)
    except Exception:
        url=HOST+'/public.php/webdav/'
        rows=listing(url)
    (out/'ibims_listing.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)
    if not a.download: return
    candidates=[r for r in rows if not r['directory'] and 'core' in r['name'].lower() and 'mat' in r['name'].lower() and r['name'].lower().endswith('.zip')]
    if not candidates: raise RuntimeError('No uniquely identified core MAT ZIP in the listing; inspect before downloading')
    selected=min(candidates,key=lambda r:r['bytes'])
    if not 0<selected['bytes']<=1024**3: raise ValueError('Core ZIP exceeds 1 GiB download limit')
    target=out/'ibims1_core_mat.zip'
    if target.exists():
        if target.stat().st_size!=selected['bytes']: raise ValueError('Existing ZIP size mismatch')
    else:
        partial=target.with_suffix('.zip.part')
        with request(urllib.parse.urljoin(HOST,selected['href'])) as src,partial.open('wb') as dst:
            count=0
            while True:
                block=src.read(1024*1024)
                if not block: break
                dst.write(block); count+=len(block)
                if count>selected['bytes']: raise ValueError('Unexpected extra bytes')
        if count!=selected['bytes']: raise ValueError('Incomplete download')
        partial.replace(target)
    h=hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    (out/'ibims_receipt.json').write_text(json.dumps(dict(source=selected,local=target.relative_to(ROOT).as_posix(),sha256=h.hexdigest()),indent=2),encoding='utf-8')


if __name__=='__main__': main()
