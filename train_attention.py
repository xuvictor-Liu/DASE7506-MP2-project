"""From-scratch attention training without a framework dependency."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import copy,pickle,json,time
from pathlib import Path
import numpy as np
from train import load
from selector_attention import build,forward
from common import parse_answer
def step_grad(x,y,valid,m,drop):
    h0=np.maximum(x@m['w1']+m['b1'],0); h=h0*drop
    key=np.where(valid,h@m['a'],-1e20); att=np.exp(key-key.max(1,keepdims=True)); att/=att.sum(1,keepdims=True)
    context=(att[:,:,None]*h).sum(1,keepdims=True); s=h@m['w2']+(h*context)@m['wc']; masked=np.where(valid,s,-1e20); p=np.exp(masked-masked.max(1,keepdims=True)); p/=p.sum(1,keepdims=True); target=y/y.sum(1,keepdims=True)
    ds=(p-target)/len(x); dh=ds[:,:,None]*(m['w2']+context*m['wc']); dc=(ds[:,:,None]*h*m['wc']).sum(1,keepdims=True)
    da=(dc*h).sum(2); dk=att*(da-(att*da).sum(1,keepdims=True)); dh+=att[:,:,None]*dc+dk[:,:,None]*m['a']
    dz=dh*drop*(h0>0)
    grads=dict(w1=x.reshape(-1,x.shape[-1]).T@dz.reshape(-1,dz.shape[-1]),b1=dz.sum((0,1)),w2=(h*ds[:,:,None]).sum((0,1)),wc=(h*context*ds[:,:,None]).sum((0,1)),a=(h*dk[:,:,None]).sum((0,1)))
    loss=-np.sum(target*np.log(np.maximum(p,1e-30)))/len(x)
    return loss,grads
def fit(raw,y,valid,seed,epochs):
    mean=raw.mean((0,1)); scale=raw.std((0,1)); scale[scale<1e-5]=1; x=np.clip((raw-mean)/scale,-10,10).astype(np.float32); rng=np.random.default_rng(seed); d=x.shape[-1]; width=64
    m=dict(w1=(rng.standard_normal((d,width))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(width,np.float32),w2=(rng.standard_normal(width)*.05).astype(np.float32),wc=(rng.standard_normal(width)*.02).astype(np.float32),a=(rng.standard_normal(width)*.05).astype(np.float32)); first={k:np.zeros_like(v) for k,v in m.items()}; second=copy.deepcopy(first); step=0; snaps=[]
    for ep in range(1,epochs+1):
        for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            drop=((rng.random((len(ids),4,width))>.2)/.8).astype(np.float32); loss,grads=step_grad(x[ids],y[ids],valid[ids],m,drop); step+=1
            for k,g in grads.items():
                if k!='b1': g=g+.002*m[k]
                g=g.astype(np.float32); first[k]=.9*first[k]+.1*g; second[k]=.999*second[k]+.001*g*g
                m[k]-=.0005*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        if ep in [20,50,100]: snaps.append((ep,dict(mean=mean,scale=scale,models=[copy.deepcopy(m)])))
    return snaps
def check_gradient():
    rng=np.random.default_rng(9); x=rng.normal(size=(2,4,3)); y=np.array([[1,0,0,0],[0,1,0,0]]); v=np.ones((2,4),bool); v[0,3]=False
    m={k:rng.normal(size=shape)*.2 for k,shape in [('w1',(3,5)),('b1',(5,)),('w2',(5,)),('wc',(5,)),('a',(5,))]}; drop=np.ones((2,4,5)); loss,g=step_grad(x,y,v,m,drop)
    for k in m:
        for index in [tuple(0 for _ in m[k].shape),tuple(n-1 for n in m[k].shape)]:
            val=m[k][index]; eps=1e-6; m[k][index]=val+eps; plus=step_grad(x,y,v,m,drop)[0]; m[k][index]=val-eps; minus=step_grad(x,y,v,m,drop)[0]; m[k][index]=val
            assert np.isclose((plus-minus)/(2*eps),g[k][index],rtol=1e-4,atol=1e-6),(k,index)
    order=[2,0,3,1]; assert np.allclose(forward(x[:,order],v[:,order],m),forward(x,v,m)[:,order]); print('Gradient and permutation checks passed.',flush=True)
def main():
    check_gradient(); start=time.perf_counter(); out=Path('runs_attention'); out.mkdir(exist_ok=True)
    b,e,_,_,_,y=load('train'); raw,v=build(b,e); hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]; records=[]
    for ep,bundle in fit(raw[tr],y[tr],v[tr],41,100):
        x=np.clip((raw[ho]-bundle['mean'])/bundle['scale'],-10,10); s=forward(x,v[ho],bundle['models'][0]); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(epoch=ep,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); members=[fit(raw[ids],y[ids],v[ids],seed,best['epoch'])[-1][1] for seed in [41,42,43]]; bundle=dict(mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members])
    bv,ev,_,_,_,yv=load('validation'); xv,vv=build(bv,ev); x=np.clip((xv-bundle['mean'])/bundle['scale'],-10,10); s=sum(forward(x,vv,m) for m in bundle['models'])/len(bundle['models']); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in row['candidates'])-{None})>=2 for row in bv]); result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
    (out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(bundle,protocol=5)); result['asset_bytes']=path.stat().st_size; (out/'selection.json').write_text(json.dumps(result,indent=2)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
