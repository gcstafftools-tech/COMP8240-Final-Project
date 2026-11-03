"""Deterministic scoring of archived model judgements; DOES NOT call a model.
To reproduce fresh LLM judgements, run the saved prompt against a disclosed model version separately.
"""
import json
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent
r=json.loads((p/'judge_records.json').read_text())
items=r['items']
# Claims have evidence-based reference labels, manually checked against public source and experiment scripts.
reference={1:'Supported',2:'Unsupported',3:'Supported',4:'Unsupported',5:'Supported',6:'Unsupported',7:'Unsupported',8:'Unverifiable',9:'Supported',10:'Unsupported'}
rows=[{'id':z['id'],'reference_label':reference[z['id']],'archived_judge_label':z['judgement'],'agreement':z['judgement']==reference[z['id']]} for z in items]
result={'evaluation':'archived output versus independently assigned evidence-based reference labels', 'n_items':len(rows),'n_agree':sum(z['agreement'] for z in rows),'agreement_fraction':sum(z['agreement'] for z in rows)/len(rows), 'reference_distribution':dict(Counter(reference.values())), 'judge_distribution':dict(Counter(z['judgement'] for z in items)),'rows':rows,'important_limit':'Scoring can be reproduced; fresh LLM generation cannot be reproduced locally without a model endpoint and prompt execution; labels are not independently collected human annotation.'}
(p/'judge_scored_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
