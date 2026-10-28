"""Candidate answer relations to quantities in the question, without solving by lookup."""
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
NUMBER_WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'dozen':12,'half':.5,'twice':2,'double':2,'triple':3}
CUES=['each','per','total','left','remain','more','less','than','times','twice','half','percent','profit','cost','sell','sold','buy','bought','hour','minute','day','week','month','year','speed','distance','average','ratio','difference','together','equally','split','share','increase','decrease','discount','tax','earn','save','spend','pair','opposite','first','second','last','rest','another','both','only','all','at least','at most','how many','how much']
def relations(bank):
    rows=[]
    for row in bank:
        q=row['question'].lower().replace(',','')
        # Ignore numeric ordinals such as grade levels.
        q=re.sub(r'\b\d+(?:st|nd|rd|th)\b','',q)
        tokens=re.findall(r'\d+(?:\.\d+)?|\.\d+|[a-z]+',q)
        values=[float(t) if re.fullmatch(r'\d+(?:\.\d+)?|\.\d+',t) else NUMBER_WORDS[t] for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?|\.\d+',t) or t in NUMBER_WORDS]
        nums=np.array(values[:12],dtype=float)
        if not len(nums): nums=np.array([0.])
        n=len(nums); a=nums[:,None]; b=nums[None,:]
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            pair=np.stack([a+b,a-b,a*b,np.broadcast_to(a,(n,n))/np.broadcast_to(b,(n,n))],axis=-1)
            pairs=pair[~np.eye(n,dtype=bool)].reshape(-1,4)
            if not len(pairs): pairs=np.zeros((1,4))+np.nan
            triples=[]
            for op in range(4):
                p=pair[:,:,op][:,:,None]; c=nums[None,None,:]
                triples += [p+c,p-c,p*c,p/c]
            tri=np.stack(triples,axis=-1)
            idx=np.indices((n,n,n)); allowed=(idx[0]!=idx[1])&(idx[0]!=idx[2])&(idx[1]!=idx[2]); tri=tri[allowed]
            if not len(tri): tri=np.zeros((1,16))+np.nan
        constants=np.array([.01,.1,.5,2,7,12,24,60,100],float)
        special=np.concatenate([nums.sum()*constants,nums.sum()/constants,nums.prod()*constants,nums.max()*constants,nums.min()*constants,
            [nums.sum(),nums.prod(),nums.mean(),nums.max()-nums.min(),abs(nums[::2].sum()-nums[1::2].sum()),nums.sum()-nums.max(),nums.max()*2-nums.sum()]])
        cues=[q.count(w) for w in CUES]
        group=[]
        for c in row['candidates']:
            answer=parse_answer(c['text']); val=float(Fraction(answer)) if answer is not None else 0.
            tol=1e-5*(1+abs(val)); pm=np.abs(pairs-val)<tol; tm=np.abs(tri-val)<tol
            f=[len(values),val<0,val==int(val),np.log1p(abs(val)),sum(abs(nums-val)<tol),val>nums.max(),val<nums.min()]
            f+=np.mean(pm,axis=0).tolist()+np.max(pm,axis=0).tolist()+np.mean(tm,axis=0).tolist()+np.max(tm,axis=0).tolist()
            f+=(np.abs(special-val)<tol).astype(float).tolist()
            f+=np.clip(val/(1+np.abs(nums[:6])), -100,100).tolist()+[0]*max(0,6-n)
            f+=cues; group.append(f)
        g=np.array(group,np.float32); rows.append(np.concatenate([g,g-g.mean(0),g-g.max(0)],1))
    return np.array(rows,np.float32)
