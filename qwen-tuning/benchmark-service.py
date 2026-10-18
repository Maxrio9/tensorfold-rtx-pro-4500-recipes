"""Portable single-stream adapter of the measured 512-output campaign harness.
Never starts or restores a model. Requires a running localhost service and tokenizers.
Raw streams are saved only to your specified local output directory.
"""
import argparse,concurrent.futures,gzip,hashlib,json,math,os,pathlib,random,statistics,threading,time,urllib.request,uuid
from tokenizers import Tokenizer
SOURCE_BYTES=pathlib.Path(__file__).read_bytes()
SOURCE_SHA=hashlib.sha256(SOURCE_BYTES).hexdigest()
def get(url,payload=None,timeout=300):
 return urllib.request.urlopen(urllib.request.Request(url,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=timeout)

def message(task,seed,body=''):
 tasktext={'code':'Write a complete Python LRU cache with get, put, delete, clear, capacity validation and extensive unittest examples. Provide at least 2000 words of code and explanations.', 'prose':'Explain how a computer handles a page fault, with detailed practical examples. Write at least 2000 words.'}[task]
 return [{'role':'user','content':f'Session {seed:016x}. '+(body+'\n\n' if body else '')+tasktext}]

def request(base,model,messages,t,limit=512,thinking=False,fixed=True,barrier=None,temperature=None,reasoning_effort=None):
 if barrier:barrier.wait(timeout=30)
 start=time.perf_counter();chunks=[];usage={};finish=None;first=last=None;firsttext=None;firstanswer=None;firstreason=None;content=[];reason=[]
 row=dict(start_monotonic=start,prompt_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest())
 deadline_s=max(1,min(300,json.loads((O/'campaign.json').read_text())['deadline_epoch']-time.time()-60))
 payload=dict(model=model,messages=messages,max_tokens=limit,temperature=(1 if thinking else .6) if temperature is None else temperature,top_p=.95,top_k=20,min_p=0,presence_penalty=0,frequency_penalty=0,seed=42,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'enable_thinking':thinking})
 if reasoning_effort is not None:payload['reasoning_effort']=reasoning_effort
 if fixed:payload['ignore_eos']=True
 try:
  with get(base+'/chat/completions',payload,timeout=deadline_s) as response:
   for line in response:
    now=time.perf_counter()
    if now-start>deadline_s:raise TimeoutError('Absolute request/campaign deadline exceeded')
    if not line.startswith(b'data:'):continue
    raw=line[5:].strip()
    if raw==b'[DONE]':break
    e=json.loads(raw)
    if e.get('error'):raise RuntimeError(str(e['error']))
    if e.get('usage'):usage=e['usage']
    for choice in e.get('choices',[]):
     delta=choice.get('delta') or {};text=delta.get('content') or '';rs=delta.get('reasoning_content') or delta.get('reasoning') or ''
     if text or rs:
      if text and firstanswer is None:firstanswer=now
      if rs and firstreason is None:firstreason=now
      if first is None:first=now;firsttext=rs+text
      last=now;chunks.append(dict(t_rel_s=now-start,content=text,reasoning=rs));content.append(text);reason.append(rs)
     if choice.get('finish_reason'):finish=choice['finish_reason']
  end=time.perf_counter();out=''.join(content);reas=''.join(reason);visible=len(t.encode(reas+out,add_special_tokens=False).ids);first_n=len(t.encode(firsttext or '',add_special_tokens=False).ids)
  # API usage can include invisible control tokens. Observable text is used for
  # post-first-content-block rate; raw chunks permit further independent audit.
  decode=(last-first) if first is not None and last is not None else None
  rate=(visible-first_n)/decode if decode and decode>0 and visible>first_n else None
  row.update(status='ok',first_answer_s=firstanswer-start if firstanswer else None,first_reasoning_s=firstreason-start if firstreason else None,end_monotonic=end,elapsed_s=end-start,ttft_s=first-start if first else None,decode_s=decode,decode_tok_s=rate,legacy_decode_tok_s=(usage.get('completion_tokens',0)-1)/decode if decode and usage.get('completion_tokens',0)>1 else None,prompt_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'),visible_output_tokens=visible,first_content_block_tokens=first_n,usage=usage,finish_reason=finish,fixed_budget_met=usage.get('completion_tokens')==limit if fixed else None,content=out,reasoning=reas,stream_chunks=len(chunks))
 except Exception as e:
  end=time.perf_counter();row.update(status='error',error=str(e),end_monotonic=end,elapsed_s=end-start,content=''.join(content),reasoning=''.join(reason))
 rawdir=O/'streams';rawdir.mkdir(parents=True,exist_ok=True);path=rawdir/(uuid.uuid4().hex+'.json.gz')
 with gzip.open(path,'wt') as f:json.dump(dict(messages=messages,payload=payload,chunks=chunks,usage=usage,finish_reason=finish),f)
 row['raw_stream']=str(path.relative_to(O));return row

def metrics(base):
 try:
  with get(base.removesuffix('/v1')+'/metrics',timeout=3) as r:text=r.read().decode()
  return [x for x in text.splitlines() if not x.startswith('#') and '_bucket' not in x and any(k in x.lower() for k in ['spec_','accept','draft','prefix_cache','preempt','request_prefill_time','num_requests_running','num_requests_waiting'])][:180]
 except Exception:return None

def burst(base,model,t,v,task,seed,mode,repeat,body='',limit=512,fixed=True,thinking=False,measured=True):
 clients=v['clients'];jobs=[message(task,seed+i*13817,body) for i in range(clients)];barrier=threading.Barrier(clients);before=metrics(base)
 with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:qs=list(pool.map(lambda m:request(base,model,m,t,limit,thinking,fixed,barrier),jobs))
 wall=max(q['end_monotonic'] for q in qs)-min(q['start_monotonic'] for q in qs)
 good=[q for q in qs if q['status']=='ok'];rates=[q['decode_tok_s'] for q in good if q.get('decode_tok_s')];ttfts=[q['ttft_s'] for q in good if q.get('ttft_s') is not None]
 def p95(xs):return sorted(xs)[min(len(xs)-1,math.ceil(.95*len(xs))-1)] if xs else None
 row=dict(bench_source_sha256=SOURCE_SHA,label=v['label'],family=v['family'],cap=v['cap'],clients=clients,tier=v.get('tier','minimal'),task=task,mode=mode,repeat=repeat,measured=measured,thinking=thinking,output_limit=limit,fixed=fixed,seed=seed,wall_s=wall,aggregate_output_tok_s=sum(q.get('output_tokens') or 0 for q in qs)/wall,aggregate_visible_output_tok_s=sum(q.get('visible_output_tokens') or 0 for q in qs)/wall,per_stream_decode_median_tok_s=statistics.median(rates) if rates else None,ttft_median_s=statistics.median(ttfts) if ttfts else None,ttft_p95_s=p95(ttfts),latency_p95_s=p95([q['elapsed_s'] for q in good]),errors=len(qs)-len(good),fixed_budget_met=sum(q.get('fixed_budget_met') is True for q in qs) if fixed else None,requests=qs,metrics_before=before,metrics_after=metrics(base),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 with (O/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps({k:x for k,x in row.items() if k not in ['requests','metrics_before','metrics_after']}),flush=True)
 if row['errors']:raise RuntimeError('Request failed; retained record, stop profile')
 return row

def main():
 global O
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base',default='http://127.0.0.1:19090/v1')
 p.add_argument('--tokenizer',required=True,type=pathlib.Path)
 p.add_argument('--out',required=True,type=pathlib.Path)
 a=p.parse_args();O=a.out
 if O.exists():p.error('--out must be a new directory; preserve earlier measurements')
 O.mkdir(parents=True)
 (O/'campaign.json').write_text(json.dumps({'deadline_epoch':time.time()+3600}))
 t=Tokenizer.from_file(str(a.tokenizer));t.no_truncation();t.no_padding()
 with get(a.base+'/models') as f:model=json.load(f)['data'][0]['id']
 v={'label':'single-service-reproduction','family':'qwen','cap':1,'clients':1,'tier':'minimal'}
 for task in ['prose','code']:
  seed=int(hashlib.sha256(f'minimal:{task}:1:warmup'.encode()).hexdigest()[:15],16)
  burst(a.base,model,t,v,task,seed,'warmup',-1,measured=False)
  for rep in range(5):
   seed=int(hashlib.sha256(f'minimal:{task}:1:{rep}'.encode()).hexdigest()[:15],16)
   row=burst(a.base,model,t,v,task,seed,'cold',rep)
   assert row['fixed_budget_met']==1
   assert all(q['usage'].get('prompt_tokens_details',{}).get('cached_tokens')==0 for q in row['requests'])
 rows=[json.loads(s) for s in (O/'measurements.jsonl').read_text().splitlines()]
 for task in ['prose','code']:
  rr=[r for r in rows if r['measured'] and r['task']==task]
  print(json.dumps({'task':task,'mean_decode_tok_s':statistics.mean(r['per_stream_decode_median_tok_s'] for r in rr),
    'mean_aggregate_tok_s':statistics.mean(r['aggregate_output_tok_s'] for r in rr),'repetitions':len(rr)}))

if __name__=='__main__':main()
