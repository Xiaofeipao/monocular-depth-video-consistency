"""Join verified archive parts for the saved relative-video reproduction bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/relative_video_demo_restored.zip')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'artifacts/relative_video_demo'
    manifest=json.loads((root/'archive_manifest.json').read_text())
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    total=hashlib.sha256()
    with output.open('xb') as stream:
        for record in manifest['parts']:
            data=(root/record['file']).read_bytes()
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise ValueError('Archive part mismatch: '+record['file'])
            total.update(data)
            stream.write(data)
    if output.stat().st_size!=manifest['bytes'] or total.hexdigest()!=manifest['sha256']:
        raise ValueError('Complete archive checksum mismatch')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP CRC mismatch')
    print('Verified ZIP:',output,'SHA256:',total.hexdigest())


if __name__=='__main__': main()
