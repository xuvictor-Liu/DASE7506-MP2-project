"""Reference-free, permutation-equivariant candidate features."""
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from arithmetic import checks
WORDS = ['wait','however','but','assuming','assume','instead','mistake','correct','therefore','total','remaining','each','per','half','percent','ratio','average','approximately','round','cannot','not','actually','check','verify','alternatively']
def features(bank):
    result, valid, agreement = [], [], []
    for row in bank:
        answers=[parse_answer(c['text']) for c in row['candidates']]
        counts=[answers.count(a) if a is not None else 0 for a in answers]
        q=row['question'].lower(); qnums=set(re.findall(r'\d+(?:\.\d+)?',q)); group=[]
        for c,a,count in zip(row['candidates'],answers,counts):
            t=c['text']; low=t.lower(); nums=re.findall(r'\d+(?:\.\d+)?',t)
            val=float(Fraction(a)) if a is not None else 0.
            f=[a is not None,count,len(set(x for x in answers if x is not None)),sum(x is not None for x in answers),len(t),len(t.split()),len(nums),len(set(nums)),len(qnums & set(nums)),len(qnums-set(nums)),np.log1p(abs(val)),val<0,val==int(val),val==0,len(q),len(qnums)]
            f += [low.count(w) for w in WORDS]
            f += [t.count(w) for w in ['\n','=','+','-','*','/','\\frac','\\times','\\boxed','####','$','**','?']]
            f += [q.count(w) for w in WORDS]
            f += checks(t)
            group.append(f)
        g=np.asarray(group,dtype=np.float32)
        result.append(np.concatenate([g,g-g.mean(0),g-g.max(0)],axis=1))
        valid.append([a is not None for a in answers]); agreement.append(counts)
    return np.asarray(result),np.asarray(valid),np.asarray(agreement)
def choose(scores,valid):
    return np.where(valid,scores,-1e20).argmax(axis=1)
