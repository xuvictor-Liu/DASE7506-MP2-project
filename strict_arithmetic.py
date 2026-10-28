"""Conservative full numeric equality extraction with variable-boundary guards."""
import re
import numpy as np
from fractions import Fraction
from features_v2 import normalize
from arithmetic import value
from common import parse_answer
def extract(text):
    t=normalize(text); checks=[]
    for match in re.finditer(r'[\d.\s()+*/=\-]+',t):
        segment=match.group(); sides=segment.split('=')
        if len(sides)<2: continue
        leading=segment[:len(segment)-len(segment.lstrip())]
        prefix=t[:match.start()].rstrip(); suffix=t[match.end():]
        incomplete_first=bool(prefix and (prefix[-1].isalnum() or prefix[-1]=='\\') and '\n' not in leading)
        incomplete_last=bool(suffix and (suffix[0].isalnum() or suffix[0]=='{'))
        for index,(left,right) in enumerate(zip(sides,sides[1:])):
            if index==0 and incomplete_first: continue
            if index==len(sides)-2 and incomplete_last: continue
            if not any(op in left+right for op in '+-*/'): continue
            try: a=value(left); b=value(right)
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): continue
            if not np.isfinite(a+b): continue
            checks.append((a,b,abs(a-b)/(1+abs(b))))
    return checks
def features(bank):
    result=[]
    for row in bank:
        answers=[parse_answer(c['text']) for c in row['candidates']]
        nums=[float(Fraction(a)) if a is not None else None for a in answers]
        eq=[extract(c['text']) for c in row['candidates']]; support=np.zeros(4); contra=np.zeros(4); group=[]
        for i,checks in enumerate(eq):
            for a,b,error in checks:
                if nums[i] is not None and error>=.01 and abs(b-nums[i])<1e-5*(1+abs(b)):
                    contra[i]+=1
                    for j,n in enumerate(nums):
                        if n is not None and abs(n-a)<1e-5*(1+abs(a)): support[j]+=1
        for i,checks in enumerate(eq):
            bad=sum(err>=.01 for _,_,err in checks); good=sum(err<1e-5 for _,_,err in checks)
            group.append([good,bad,len(checks),bad/(1+len(checks)),max([min(err,10) for _,_,err in checks],default=0),min(support[i],2),min(contra[i],2),support[i]>0,contra[i]>0])
        g=np.array(group,np.float32); result.append(np.concatenate([g,g-g.mean(0),g-g.max(0)],1))
    return np.array(result,np.float32)
