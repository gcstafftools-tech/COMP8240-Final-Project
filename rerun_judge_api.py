"""Optional reproducible LLM-as-judge rerun using a model API.

Needs OPENAI_API_KEY and JUDGE_MODEL environment variables and paid/eligible API
access. This script was NOT executed in producing the archived conversations.
Records precise model and response for every claim; never overwrites original.

Example: JUDGE_MODEL=<accessible-model> OPENAI_API_KEY=<secret> python rerun_judge_api.py
"""
import os, json, urllib.request, datetime, pathlib
base=pathlib.Path(__file__).parent
key=os.environ.get('OPENAI_API_KEY'); model=os.environ.get('JUDGE_MODEL')
if not key or not model: raise SystemExit('Set OPENAI_API_KEY and JUDGE_MODEL. Never commit your key.')
claims=json.loads((base/'judge_records.json').read_text())['items']
rubric=(base/'JUDGE_PROMPT.txt').read_text()
records=[]
for row in claims:
    prompt=f'{rubric}\n\nClaim ID: {row["id"]}\nClaim: {row["claim"]}\nAnswer ONLY with Supported, Unsupported, or Unverifiable, followed by one concise reason.'
    body={'model':model,'input':prompt}
    req=urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(body).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=120) as resp: ans=json.loads(resp.read())
    records.append({'id':row['id'],'claim':row['claim'],'model':model,'request_prompt':prompt,'api_response_id':ans.get('id'),'raw_output':ans.get('output'),'output_text':ans.get('output_text')})
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=base/f'judge_live_run_{stamp}.json'
out.write_text(json.dumps({'kind':'actual external API responses','timestamp_utc':stamp,'model':model,'records':records},indent=2))
print(out)
