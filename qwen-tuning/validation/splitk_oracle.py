"""Real-weight affine4 group-kernel oracle, paired projections and pipeline/row-tile comparisons.
Bit comparisons retain K splits; FP64 product is independent of the CUDA group kernel.
"""
import ctypes as C,hashlib,json,pathlib,struct,time
import torch
from cuda.bindings import driver as cu
torch.set_num_threads(4)
O=pathlib.Path('/results');B=pathlib.Path('/models/qwen-tf-native-4bit');root=pathlib.Path('/src/.zig-cache')
marker=b'ILi64ELi16ELi64ELi1ELi4ELi2E';fatbins=[p for p in root.rglob('qmm_group.fatbin') if marker in p.read_bytes()];assert len(fatbins)==1,fatbins
packbins=list(root.rglob('affine4_pack.fatbin'));assert len(packbins)==1,packbins
lib=C.CDLL('libcuda.so.1');torch.cuda.init();anchor=torch.empty(1,device='cuda')
def check(rc):
 if rc:
  msg=C.c_char_p();lib.cuGetErrorString(rc,C.byref(msg));raise RuntimeError((rc,msg.value))
def mod(path):
 v=C.c_void_p();lib.cuModuleLoad.argtypes=[C.POINTER(C.c_void_p),C.c_char_p];check(lib.cuModuleLoad(C.byref(v),str(path).encode()));return v
def fn(module,name,shared=0):
 v=C.c_void_p();lib.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p];check(lib.cuModuleGetFunction(C.byref(v),module,name.encode()))
 if shared:lib.cuFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int];check(lib.cuFuncSetAttribute(v,8,shared))
 return v
m=mod(fatbins[0]);pm=mod(packbins[0]);pack=fn(pm,'tf_pack_dense')
variants=[('s2',16,64,1,4,2,8832),('s3',16,64,1,4,3,13248),('m64s2',64,64,1,4,2,21504),('m64s3',64,64,1,4,3,32256),('m64n128s2',64,128,2,4,2,32768),('s8',16,64,1,4,8,35328),('s4',16,64,1,4,4,17664),('s6',16,64,1,4,6,26496),('m32',32,64,1,4,4,26112),('m64n64',64,64,1,4,4,43008),('m64n128',64,128,2,4,3,39168),('m128n64',128,64,2,4,3,57600),('m128n128',128,128,2,4,2,65536)]
funcs={}
for label,bm,bn,wm,wn,st,smem in variants:
 symbol=f'_ZN12tf_qmm_group12group_kernelILi64ELi{bm}ELi{bn}ELi{wm}ELi{wn}ELi{st}ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii'
 funcs[label]=fn(m,symbol,smem)
lib.cuLaunchKernel.argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
lib.cuLaunchKernelEx.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
def ptr(t):return C.c_void_p(t.data_ptr())
def args(values):return (C.c_void_p*len(values))(*[C.cast(C.pointer(v),C.c_void_p) for v in values])
def launch_pack(values,grid):check(lib.cuLaunchKernel(pack,grid,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),args(values),None))
class Part(C.Structure):_fields_=[(k,C.c_uint64) for k in ['w','scales','biases','out']]+[(k,C.c_int) for k in ['n','npad','sk','tiles','first']]
class Parts(C.Structure):_fields_=[('p',Part*4),('count',C.c_int)]
assert C.sizeof(Part)==56 and C.sizeof(Parts)==232
index={}
for file in B.glob('*.safetensors'):
 with file.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n))
 for key,info in head.items():
  if key!='__metadata__':index[key]=(file,8+n,info)
def tensor(key):
 file,offset,d=index[key];a,b=d['data_offsets']
 with file.open('rb') as f:f.seek(offset+a);raw=bytearray(f.read(b-a))
 return torch.frombuffer(raw,dtype={'U32':torch.int32,'BF16':torch.bfloat16}[d['dtype']]).reshape(d['shape'])

# New reduction order is not required to be bit-identical to the original. Bounds were declared before collection.
plan=json.loads(pathlib.Path('/study/outputs/qwen-profile/splitk-numerical-plan.json').read_text())
results=[];t0=time.time();flush=torch.zeros(32*1024*1024,dtype=torch.float32,device='cuda')
classes=['language_model.model.layers.0.mlp.down_proj','language_model.model.layers.0.linear_attn.in_proj_qkv','language_model.model.layers.0.linear_attn.out_proj','language_model.model.layers.3.self_attn.o_proj']
for name in classes:
 words=tensor(name+'.weight');sc=tensor(name+'.scales');bias=tensor(name+'.biases');n,k8=words.shape;k=k8*8;kg=k//64;npad=(n+127)//128*128;assert n==npad
 codes=((words.to(torch.int64).unsqueeze(-1)>>(4*torch.arange(8,dtype=torch.int64)))&15).reshape(n,kg,64)
 reference_w=(codes.double()*sc.double().unsqueeze(-1)+bias.double().unsqueeze(-1)).reshape(n,k).cuda();del codes
 w=words.cuda();qw=torch.empty(npad*k//8,dtype=torch.int32,device='cuda');qs=sc.T.contiguous().cuda();qb=bias.T.contiguous().cuda()
 launch_pack([ptr(w),C.c_int(n),C.c_int(k8),C.c_int(kg),ptr(qw),C.c_longlong(qw.numel())],(qw.numel()+255)//256);torch.cuda.synchronize();del w,words,sc,bias
 for rows in [1,8,16,64,128]:
  x=torch.randn((rows,k),generator=torch.Generator().manual_seed(4628+rows)).to(torch.bfloat16).cuda();xs=x.float().reshape(rows,kg,64).sum(-1).contiguous();reference=x.double()@reference_w.T;outs={};errs={};timings={};splits={}
  labels=['s8','s4'] if rows<=16 else ['m64n64'] if rows<=64 else ['m64n128']
  for threshold in [192,128]:
   sk=1
   while sk<8 and ((n+63)//64)*sk<threshold and kg%(sk*2)==0 and kg//(sk*2)>=8:sk*=2
   for label in labels:
    _,bm,bn,wm,wn,st,smem=next(v for v in variants if v[0]==label);rt=(rows+bm-1)//bm;tiles=(n+bn-1)//bn;out=torch.empty((rows,n),dtype=torch.bfloat16,device='cuda');parts=Parts();parts.count=1;parts.p[0]=Part(qw.data_ptr(),qs.data_ptr(),qb.data_ptr(),out.data_ptr(),n,npad,sk,tiles,0)
    values=[ptr(x),ptr(xs),parts]+[C.c_int(v) for v in [rows,k,k,rt,sk]];aa=args(values)
    cfg=cu.CUlaunchConfig();cfg.gridDimX=rt*tiles*sk;cfg.gridDimY=cfg.gridDimZ=1;cfg.blockDimX=wm*wn*32;cfg.blockDimY=cfg.blockDimZ=1;cfg.sharedMemBytes=smem;cfg.hStream=torch.cuda.current_stream().cuda_stream
    if sk>1:
     attr=cu.CUlaunchAttribute();attr.id=cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;attr.value.clusterDim.x=sk;attr.value.clusterDim.y=attr.value.clusterDim.z=1;cfg.attrs=[attr];cfg.numAttrs=1
    def go():check(lib.cuLaunchKernelEx(C.c_void_p(cfg.getPtr()),funcs[label],aa,None))
    go();torch.cuda.synchronize();key=str(threshold)+'-'+label;outs[key]=out;errs[key]=(out.double()-reference).abs();splits[key]=sk;times=[]
    for rep in range(10):
     flush.add_(1);b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);b.record();go();e.record();e.synchronize();times.append(b.elapsed_time(e)*1000)
    timings[key]=sum(times)/len(times)
  record={'projection':name,'shape':[rows,n,k],'candidates':{}}
  for key,out in outs.items():
   label=key.split('-',1)[1];base='192-'+label;er=errs[key];be=errs[base];mismatches=int((out.view(torch.int16)!=outs[base].view(torch.int16)).sum());fraction=mismatches/out.numel();ma=float(er.max());bma=float(be.max());p99=float(torch.quantile(er.flatten(),.99));bp99=float(torch.quantile(be.flatten(),.99));finite=bool(torch.isfinite(out).all());passed=finite and ma<=bma*plan['maximum_error_ratio']+plan['absolute_slack'] and p99<=bp99*plan['p99_error_ratio']+plan['absolute_slack'] and fraction<=plan['maximum_bit_mismatch_fraction']
   record['candidates'][key]={'split_k':splits[key],'bit_mismatches':mismatches,'bit_mismatch_fraction':fraction,'max_abs_error':ma,'baseline_max_abs_error':bma,'p99_abs_error':p99,'baseline_p99_abs_error':bp99,'cold_microbench_us':timings[key],'finite':finite,'passed':passed}
   assert passed,record
  results.append(record);print(json.dumps(record),flush=True);del x,xs,reference,outs,errs
 del qw,qs,qb,reference_w;torch.cuda.empty_cache()
receipt={'all_passed':True,'results':results,'elapsed_s':time.time()-t0,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'fatbin_sha256':hashlib.sha256(fatbins[0].read_bytes()).hexdigest(),'reference':'Real affine4 tensors decoded in FP64; full-K FP64 product. Shape-fixed split reduction changes may alter FP32 rounding, so predeclared numeric bounds replace exact-baseline bits. Every timed launch follows a128MiB read/write cache-pollution pass; no claim of complete cache eviction. New quality/parity qualification still required.'}
(O/'splitk-oracle.json').write_text(json.dumps(receipt,indent=2)+'\n')
