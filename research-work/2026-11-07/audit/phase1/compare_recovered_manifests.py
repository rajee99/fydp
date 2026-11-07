"""Compare outputs of successful existing scripts with preserved baseline rows."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json
AUDIT=Path(__file__).resolve().parent;ROOT=AUDIT.parents[3]
OUT=AUDIT/'recovery_attempt_02';BOX=OUT/'sandbox'
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
comparisons=[]
for name in ('target_manifest.csv','eval_manifest.csv','test_set_v2_candidates_to_download.csv'):
    a=ROOT/'data/processed'/name;b=BOX/'data/processed'/name
    ar,br=rows(a),rows(b)
    def key(r):return (r['video_id'],r.get('start_seconds'),r.get('end_seconds'),r.get('target_class'))
    aset,bset={key(r) for r in ar},{key(r) for r in br}
    ae={key(r) for r in ar if r.get('target_class')!='other'}
    be={key(r) for r in br if r.get('target_class')!='other'}
    comparisons.append({'file':name,'baseline_rows':len(ar),'generated_rows':len(br),
        'baseline_class_counts':dict(Counter(r.get('target_class') for r in ar)),
        'generated_class_counts':dict(Counter(r.get('target_class') for r in br)),
        'baseline_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),
        'parsed_rows_identical':ar==br,'shared_segment_class_keys':len(aset&bset),
        'baseline_only_keys':len(aset-bset),'generated_only_keys':len(bset-aset),
        'emergency_segment_keys_equal':ae==be})
(OUT/'manifest_comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
print(json.dumps(comparisons,indent=2))
