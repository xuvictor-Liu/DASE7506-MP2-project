"""Question-level reliability features for two experts; inputs only."""
import numpy as np
from fractions import Fraction
from common import parse_answer
from strict_arithmetic import extract
from selector_semantic_blend import norm
def build(bank,semantic,cnn,valid):
    predictions=[np.where(valid,s,-1e20).argmax(1) for s in [semantic,cnn]]
    normalized=[norm(s,valid) for s in [semantic,cnn]]; result=[]
    for i,row in enumerate(bank):
        answers=[parse_answer(c['text']) for c in row['candidates']]; vectors=[]
        for j in [0,1]:
            p=predictions[j][i]; other=predictions[1-j][i]; raw=[semantic,cnn][j][i]; z=normalized[j][i]; probs=np.exp(np.where(valid[i],z,-1e20)-z[valid[i]].max()); probs/=probs.sum(); ranked=np.sort(z[valid[i]])
            text=row['candidates'][p]['text']; eq=extract(text); answer=answers[p]; value=float(Fraction(answer)) if answer is not None else 0
            vectors.append([ranked[-1]-ranked[-2] if len(ranked)>1 else 0,z[p]-z[other],float(probs[p]),float(probs[other]),float(-(probs*np.log(np.maximum(probs,1e-12))).sum()),float(np.std(raw[valid[i]])),answers.count(answer)/4,np.log1p(len(text)),len(eq),sum(err>.001 for _,_,err in eq),np.log1p(abs(value)),sum(text.lower().count(w) for w in ['wait','however','instead','mistake','actually'])])
        a=np.asarray(vectors[0]); b=np.asarray(vectors[1]); result.append(np.r_[a,b,b-a])
    return np.nan_to_num(np.asarray(result,np.float32),nan=0,posinf=100,neginf=-100)
