"""Attempt two exact source segments through the unchanged existing downloader."""
from pathlib import Path
import concurrent.futures, hashlib, importlib.metadata, importlib.util, json, os, sys, urllib.request

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[3]
OUT=AUDIT/'recovery_attempt_02'
SANDBOX=OUT/'sandbox'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
sys.path.insert(0,str(SANDBOX/'src'))
from batch_downloader import download_single_segment
import soundfile as sf
records=[('7vfZM3gDjBc',480,490,'data/raw/targeted_expansion/other/other_7vfZM3gDjBc_480_490.wav'),
         ('0H2uMhzSitY',520,530,'data/raw/train/glass_breaking/glass_breaking_0H2uMhzSitY_520_530.wav')]
def attempt(record):
    vid,start,end,rel=record; path=SANDBOX/rel
    ok,message,attempts=download_single_segment(vid,start,end,path,max_retries=1)
    result={'source_url':f'https://www.youtube.com/watch?v={vid}','start':start,'end':end,'output':rel,
            'success':ok,'error':message,'attempts':attempts,'method':'unchanged batch_downloader.download_single_segment',
            'yt_dlp_version':importlib.metadata.version('yt-dlp')}
    if path.exists():
        result['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            info=sf.info(path); result['audio_info']={'sample_rate':info.samplerate,'channels':info.channels,'subtype':info.subtype,'duration':info.duration}
        except Exception as exc: result['audio_error']=repr(exc)
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    results=list(executor.map(attempt,records))
(OUT/'audio_recovery_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
try:
    request=urllib.request.Request('https://zenodo.org/records/2535878/files/NIGENS.zip?download=1',headers={'Range':'bytes=0-31'})
    with urllib.request.urlopen(request,timeout=30) as response:
        metadata={'url':response.url,'status':response.status,'headers':dict(response.headers),'prefix_hex':response.read(32).hex(),
                  'note':'Range availability check only; archive not recovered'}
except Exception as exc: metadata={'error':repr(exc)}
(OUT/'nigens_source_probe.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
