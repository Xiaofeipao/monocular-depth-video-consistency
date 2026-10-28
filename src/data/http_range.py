"""Read small official HDF5/ZIP subsets without downloading multi-GB archives.

Curl verifies TLS using the platform trust store. Cached ranges are verified
against exact Content-Range lengths; a server ignoring Range is rejected.
"""
import hashlib
import io
from pathlib import Path
import re
import shutil
import subprocess


def curl_executable():
    executable=shutil.which('curl.exe') or shutil.which('curl')
    if executable is None:
        raise FileNotFoundError('curl with verified TLS and byte-range support is required')
    return executable


def curl_download(url,path,maximum_bytes=8*1024*1024):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([curl_executable(),'--fail','--location','--silent','--show-error','--retry','2',
        '--connect-timeout','20','--max-time','120','--max-filesize',str(maximum_bytes),
        '--output',str(path),url],check=True)
    return path


class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size,cache,block_size=1024*1024,byte_budget=256*1024*1024):
        super().__init__()
        self.url,self.size,self.block_size=url,int(size),int(block_size)
        self.position,self.downloaded=0,0
        self.byte_budget=byte_budget
        self.cache=Path(cache)/hashlib.sha256(url.encode()).hexdigest()[:16]
        self.cache.mkdir(parents=True,exist_ok=True)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if position<0:
            raise ValueError('Invalid remote seek')
        self.position=int(position)
        return self.position

    def _block(self,index):
        start=index*self.block_size
        end=min(self.size,start+self.block_size)-1
        path=self.cache/f'{start}-{end}.bin'
        header=path.with_suffix('.headers')
        expected=end-start+1
        if not path.exists() or path.stat().st_size!=expected or not header.exists():
            if self.downloaded+expected>self.byte_budget:
                raise ValueError('Remote subset byte budget exceeded; full download not attempted')
            subprocess.run([curl_executable(),'--fail','--location','--silent','--show-error','--retry','2',
                '--connect-timeout','20','--max-time','90','--range',f'{start}-{end}',
                '--max-filesize',str(expected),'--dump-header',str(header),'--output',str(path),self.url],check=True)
            self.downloaded+=expected
            print(f'Fetched official source range {start}-{end}',flush=True)
        headers=header.read_text(encoding='latin1')
        content_ranges=re.findall(r'Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)',headers,re.I)
        if not content_ranges or tuple(map(int,content_ranges[-1]))!=(start,end,self.size) or path.stat().st_size!=expected:
            raise ValueError('Server did not honor the exact byte range/source size')
        return path.read_bytes()

    def read(self,size=-1):
        if size is None or size<0:
            size=self.size-self.position
        size=min(size,max(0,self.size-self.position))
        chunks=[]
        while size>0:
            index,offset=divmod(self.position,self.block_size)
            block=self._block(index)
            take=min(size,len(block)-offset)
            chunks.append(block[offset:offset+take])
            self.position+=take
            size-=take
        return b''.join(chunks)

    def readinto(self,buffer):
        data=self.read(len(buffer))
        buffer[:len(data)]=data
        return len(data)
