// CPU-only validation; no network or publication. Uses the captured official website validator.
import {readFileSync} from 'node:fs';
import {validateReceipt} from './official-protocol.mjs';
const file=process.argv[2];
if(!file)throw new Error('Usage: node validate_receipt.mjs path/to/benchmark.json');
const data=readFileSync(file);
if(data.length>1024*1024)throw new Error('Receipt exceeds the website 1 MiB limit');
const receipt=validateReceipt(JSON.parse(data));
const settings=receipt.settings;
const problems=[];
if(receipt.suite.id!=='community-v1')problems.push('Wrong suite');
if(!settings.managed||settings.rank_count!==1)problems.push('Needs managed single-rank execution');
if(settings.tokens!==256||settings.repetitions!==5)problems.push('Needs standard output/repetition counts');
if(JSON.stringify([...settings.temperatures].sort())!=='[0,1]')problems.push('Needs greedy and sampled cells');
if(receipt.model.local||!receipt.model.repo_id||!receipt.model.revision)problems.push('Missing public pinned target');
if(!receipt.hardware.gpus.length)problems.push('Missing GPU hardware');
for(const f of ['code','chat'])for(const t of [1,0]){
 const samples=receipt.samples.filter(s=>s.fixture_id===f&&s.temperature===t&&s.repeat>=0);
 if(samples.length!==5||samples.some(s=>s.status!=='ok'))problems.push(`Incomplete ${f} T=${t}`);
 if(samples.some(s=>s.cached_tokens!==0))problems.push(`Nonzero or unknown cache reuse ${f} T=${t}`);
}
console.log(JSON.stringify({schema_valid:true,standard_candidate:problems.length===0,problems,publication:'Deferred. This command never uploads.',model:receipt.model.repo_id,gpus:receipt.hardware.gpus.map(g=>g.name)},null,2));
if(problems.length)process.exitCode=2;
