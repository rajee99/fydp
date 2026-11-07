"""Read official ZIP by HTTP ranges; recover fold metadata and required waveforms."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, csv, hashlib, io, json, struct, time, urllib.request, zipfile, zlib

AUDIT=Path(__file__).resolve().parent; ROOT=AUDIT.parents[3]
OUT=AUDIT/'recovery_attempt_02'; DEST=OUT/'sandbox/NIGENS'
URL='https://zenodo.org/records/2535878/files/NIGENS.zip?download=1'
SIZE=2174918847
requests=[]
def fetch(start,end):
    for attempt in range(3):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206: raise RuntimeError('Server did not honor range request')
                data=response.read()
                if len(data)!=end-start+1: raise RuntimeError('Incorrect range length')
                return data
        except Exception:
            if attempt==2:raise
            time.sleep(3*(attempt+1))
class Remote(io.RawIOBase):
    def __init__(self): self.position=0
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else SIZE+offset
        return self.position
    def read(self,size=-1):
        size=SIZE-self.position if size<0 else min(size,SIZE-self.position)
        if not size:return b''
        data=fetch(self.position,self.position+size-1); self.position+=len(data); return data
with zipfile.ZipFile(Remote()) as archive:
    entries={info.filename:info for info in archive.infolist()}
inventory=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc':i.CRC} for i in entries.values()]
(OUT/'nigens_archive_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
with (ROOT/'data/processed/combined_training_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
    required={r['file_path'].replace('\\','/') for r in csv.DictReader(stream) if r['source_dataset']=='NIGENS'}
chosen=[i for i in entries.values() if i.filename in required or i.filename.endswith('.flist')]
found={i.filename for i in chosen}
missing=sorted(required-found)
parser=argparse.ArgumentParser()
parser.add_argument('--reverse',action='store_true')
args=parser.parse_args()
register=OUT/('nigens_reverse_recovery_register.json' if args.reverse else 'nigens_recovery_register.json')
results=json.loads(register.read_text(encoding='utf-8'))['files'] if register.exists() else []
done={r['name'] for r in results if r.get('success')}
def extract(info):
    try:
        relative=info.filename.removeprefix('NIGENS/')
        if '..' in relative.split('/') or relative.startswith('/'):raise ValueError('Unsafe archive path')
        path=DEST/relative
        if not path.exists():
            # Read fixed local header, then compressed data; central directory supplies CRC and length.
            header=fetch(info.header_offset,info.header_offset+29)
            fields=struct.unpack('<4s5H3I2H',header)
            start=info.header_offset+30+fields[-2]+fields[-1]
            blob=fetch(start,start+info.compress_size-1) if info.compress_size else b''
            if info.compress_type==zipfile.ZIP_STORED:data=blob
            else:data=zipfile._get_decompressor(info.compress_type).decompress(blob)
            if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('ZIP CRC mismatch')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(data)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        return {'name':info.filename,'success':True,'bytes':path.stat().st_size,'sha256':digest,'zip_crc':info.CRC}
    except Exception as exc:return {'name':info.filename,'success':False,'error':repr(exc)}
pending=sorted((i for i in chosen if i.filename not in done),key=lambda i:(i.filename.endswith('.wav'),i.filename))
if args.reverse:pending.reverse()
with ThreadPoolExecutor(max_workers=4 if args.reverse else 2) as executor:
    for n in range(0,len(pending),4):
        started=time.time(); batch=list(executor.map(extract,pending[n:n+4]));results.extend(batch)
        register.write_text(json.dumps({'source_url':URL,'source_record':'2535878 v1','archive_bytes':SIZE,
            'required_waveforms':len(required),'missing_archive_members':missing,'files':results},indent=2),encoding='utf-8')
        print('NIGENS recovered',sum(r['success'] for r in results),'of',len(chosen),'selected files',flush=True)
        if any(not r['success'] for r in batch):break
        # Stay below the server's observed request-rate limit; two requests per file.
        time.sleep(max(0,(7 if args.reverse else 5)-(time.time()-started)))
