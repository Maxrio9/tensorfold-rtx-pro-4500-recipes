"""Run only inside a GPU-free isolated container. No network, user files, credentials or Docker socket."""
import argparse,ast,copy,hashlib,json,pathlib,re,time,traceback
from evalplus.eval import untrusted_check,PASS
from evalplus.gen.util import trusted_exec
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--input',action='append',default=[]);p.add_argument('--self-check',action='store_true');a=p.parse_args()
problems={d['task_id']:d for d in [json.loads(x) for x in pathlib.Path(a.dataset).read_text().splitlines() if x]};oracle={};reports=[]
for id,d in problems.items():
 code=d['prompt']+d['canonical_solution'];entry={}
 for kind in ['base','plus']:
  values,times=trusted_exec(code,d[kind+'_input'],d['entry_point'],record_time=True)
  entry[kind]={'values':values,'times':times}
 oracle[id]=entry
print(json.dumps({'event':'oracles_ready','tasks':len(problems)}),flush=True)
def check(d,code,kind):
 o=oracle[d['task_id']][kind]
 status,details=untrusted_check('humaneval',code,d[kind+'_input'],d['entry_point'],expected=o['values'],atol=d['atol'],ref_time=o['times'],fast_check=True)
 return {'status':status,'passed':status==PASS,'inputs':len(d[kind+'_input']),'checked_before_stop':len(details)}
if a.self_check:
 checks=[]
 for d in problems.values():
  result=check(d,d['prompt']+d['canonical_solution'],'plus');checks.append({'id':d['task_id'],**result})
 print(json.dumps({'event':'self_check','all_passed':all(x['passed'] for x in checks),'checks':checks}),flush=True)
 unexpected=[x for x in checks if not x['passed'] and x['id']!='HumanEval/32']
 assert not unexpected,('Unexpected reference self-check failure',unexpected)
for file in a.input:
 rows=[json.loads(x) for x in pathlib.Path(file).read_text().splitlines() if x];results=[]
 for r in rows:
  if r['category']!='humaneval_plus':continue
  d=problems[r['id']];text=r['request'].get('content','');blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S);code='\n\n'.join(blocks) if blocks else text
  # Fixed parser: fenced Python blocks concatenate in source order, otherwise exact raw response.
  try:
   ast.parse(code);base=check(d,code,'base');plus=check(d,code,'plus') if base['passed'] else {'passed':False,'status':'base_failed','inputs':len(d['plus_input'])}
  except Exception as e:base={'passed':False,'status':'invalid_python','error':str(e)};plus={'passed':False,'status':'invalid_python'}
  row={'id':d['task_id'],'base':base,'plus':plus,'finish_reason':r['request'].get('finish_reason'),'code_sha256':hashlib.sha256(code.encode()).hexdigest()};results.append(row)
  print(json.dumps({'event':'task','input':file,**row}),flush=True)
 expected_count=sum(r['category']=='humaneval_plus' for r in rows) if any(r.get('supplemental_retest') or r.get('all_problems') for r in rows) else 32 if any(r.get('thinking') for r in rows) else 164
 summary={'input':file,'tasks':len(results),'expected_tasks':expected_count,'complete':len(results)==expected_count,'base_passed':sum(x['base']['passed'] for x in results),'plus_passed':sum(x['plus']['passed'] for x in results),'plus_passed_excluding_reference_exception32':sum(x['plus']['passed'] for x in results if x['id']!='HumanEval/32'),'denominator_excluding_reference_exception32':expected_count-int(any(x['id']=='HumanEval/32' for x in results)),'reference_exception':'HumanEval/32 canonical Newton solver fails an official residual test; tests unchanged','truncated':sum(x['finish_reason']=='length' for x in results),'results':results,'parser':'Concatenate fenced Python blocks, else exact raw response; no task-specific repairs','execution':'Official EvalPlus untrusted_check; same full v0.1.10 tests and default timing factors'}
 reports.append(summary);print(json.dumps({'event':'model_summary',**summary}),flush=True)
print(json.dumps({'event':'complete','reports':reports}),flush=True)
