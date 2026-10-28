"""Generic final-equation contradiction and correction evidence."""
import re
import numpy as np
from fractions import Fraction
from arithmetic import value
from common import parse_answer
from features_v2 import normalize
def evidence(bank):
    result=[]
    for row in bank:
        aa=[parse_answer(c['text']) for c in row['candidates']]
        nums=[float(Fraction(a)) if a is not None else None for a in aa]
        support=np.zeros(4); contradiction=np.zeros(4)
        for i,c in enumerate(row['candidates']):
            if nums[i] is None: continue
            for segment in re.findall(r'[\d.\s()+*/=\-]+',normalize(c['text'])):
                if '=' not in segment: continue
                sides=segment.split('=')
                for left,right in zip(sides,sides[1:]):
                    if not any(op in left for op in '+-*/'): continue
                    try: a=value(left); b=value(right)
                    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): continue
                    if not np.isfinite(a+b): continue
                    if abs(b-nums[i])>1e-5*(1+abs(b)) or abs(a-b)<.01*(1+abs(b)): continue
                    matches=[j for j,n in enumerate(nums) if n is not None and abs(n-a)<1e-5*(1+abs(a))]
                    if matches:
                        contradiction[i]+=1
                        for j in matches: support[j]+=1
        result.append(np.minimum(support,1)-np.minimum(contradiction,1))
    return np.asarray(result)
