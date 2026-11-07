"""Recover required source segments in bounded batches; stop on unresolved failure.

Calls the existing downloader unchanged. A failure blocks complete preparation,
so preserve the queue and evidence rather than bulk downloading past that gate.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, importlib.metadata, json, os, sys, time
import soundfile as sf

AUDIT=Path(__file__).resolve().parent; ROOT=AUDIT.parents[3]
OUT=AUDIT/'recovery_attempt_02'; SANDBOX=OUT/'sandbox'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
sys.path.insert(0,str(SANDBOX/'src'))
from batch_downloader import download_single_segment
def load(name):
    with (ROOT/'data/processed'/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
jobs=[]
for r in load('targeted_expansion_review_results.csv'):
    if (r['target_class']=='other' and r['verdict']=='ACCEPT') or r['verdict']=='RECLASSIFY':
        jobs.append({'video_id':r['video_id'],'start':float(r['start_seconds']),'end':float(r['end_seconds']),
                     'path':r['file_path'],'pool':'reviewed_targeted','baseline_sha256':None})
for r in load('combined_training_manifest.csv'):
    if r['source_dataset']=='AudioSet':
        jobs.append({'video_id':r['video_id'],'start':float(r['start_seconds']),'end':float(r['end_seconds']),
                     'path':r['file_path'],'pool':'baseline_train','baseline_sha256':r['file_hash']})
assert sum(j['pool']=='reviewed_targeted' for j in jobs)==224
assert sum(j['pool']=='baseline_train' for j in jobs)==527
(OUT/'required_audio_queue.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
register=OUT/'required_audio_recovery.json'
results=json.loads(register.read_text(encoding='utf-8')) if register.exists() else []
done={r['path'] for r in results if r['success']}
def attempt(job):
    result=dict(job); p=SANDBOX/job['path']; started=time.time()
    if p.exists(): ok=True; message='existing recovered segment'; n=0
    else: ok,message,n=download_single_segment(job['video_id'],job['start'],job['end'],p,max_retries=2)
    result.update(success=ok,error=message,attempts=n,elapsed_seconds=round(time.time()-started,2))
    if ok:
        info=sf.info(p)
        valid=info.samplerate==16000 and info.channels==1 and info.subtype=='PCM_16' and info.duration>0
        result.update(success=valid,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),sample_rate=info.samplerate,
                      channels=info.channels,subtype=info.subtype,duration=info.duration)
        if job['baseline_sha256']: result['byte_identical_to_original']=result['sha256']==job['baseline_sha256']
    return result
pending=[j for j in jobs if j['path'] not in done]
with ThreadPoolExecutor(max_workers=8) as executor:
    for start in range(0,len(pending),8):
        batch=list(executor.map(attempt,pending[start:start+8]))
        results.extend(batch); register.write_text(json.dumps(results,indent=2),encoding='utf-8')
        print('Recovered',sum(r['success'] for r in results),'of',len(jobs),'required segments',flush=True)
        failures=[r for r in batch if not r['success']]
        if failures:
            print('Stopping on unresolved source failures:',json.dumps(failures),flush=True)
            break
