"""Expanded equation verification and cross-candidate evidence, no references."""
import re
from fractions import Fraction
import numpy as np
from arithmetic import value
from common import parse_answer
from features import features

def normalize(text):
    t=text.replace(',','').replace('−','-').replace('×','*').replace('÷','/')
    for a,b in [('\\times','*'),('\\cdot','*'),('\\div','/'),('\\left',''),('\\right',''),('\\,',''),('\\!',''),('\\$',''),('$',''),('**','')]: t=t.replace(a,b)
    for _ in range(5): t=re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}',r'((\1)/(\2))',t)
    t=re.sub(r'\\(?:text|mathrm|mathbf|boxed)\{([^{}]*)\}',r' \1 ',t)
    t=t.replace('\\%','%')
    t=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',t)
    return t

def equations(text):
    t=normalize(text)
    good=bad=near=0; errors=[]; outputs=[]
    # Only fully numeric adjacent sides; every equality in a chain is tested.
    for segment in re.findall(r'[\d.\s()+*/=\-]+',t):
        if '=' not in segment: continue
        sides=segment.split('=')
        for left,right in zip(sides,sides[1:]):
            left=left.strip(); right=right.strip()
            if not left or not right: continue
            if not any(c in left+right for c in '+-*/'): continue
            try:
                a=value(left); b=value(right)
                if not np.isfinite(a+b): continue
                err=abs(a-b)/(1+abs(b)); outputs.append(b)
                good+=err<1e-5; near+=err<.01; bad+=err>=.01; errors.append(min(err,100))
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): pass
    return [good,bad,near,len(errors),bad/(1+len(errors)),sum(errors)/(1+len(errors)),max(errors,default=0)],outputs

def features_v2(bank, embeddings):
    base,valid,votes=features(bank); extra=[]
    for row in bank:
        texts=[c['text'] for c in row['candidates']]
        answers=[parse_answer(t) for t in texts]
        eq=[equations(t) for t in texts]
        nums=[set(re.findall(r'\d+(?:\.\d+)?',t.replace(',',''))) for t in texts]
        words=[set(re.findall(r'[a-z]+',t.lower())) for t in texts]
        group=[]
        for i,t in enumerate(texts):
            a=float(Fraction(answers[i])) if answers[i] is not None else 0
            other=[j for j in range(4) if j!=i]
            number_j=[len(nums[i]&nums[j])/max(1,len(nums[i]|nums[j])) for j in other]
            word_j=[len(words[i]&words[j])/max(1,len(words[i]|words[j])) for j in other]
            support=[sum(abs(a-b)<1e-5*(1+abs(a)) for b in eq[j][1]) for j in range(4)]
            f=eq[i][0]+[sum(number_j)/3,max(number_j),min(number_j),sum(word_j)/3,max(word_j),min(word_j),support[i],sum(support),sum(support[j]>0 for j in other)]
            # Compare evidence within the same final-answer group and against rivals.
            same=[j for j in range(4) if answers[i] is not None and answers[j]==answers[i]]
            diff=[j for j in range(4) if answers[j]!=answers[i]]
            for js in [same,diff]:
                f += [sum(eq[j][0][k] for j in js)/max(1,len(js)) for k in [0,1,4]]
            group.append(f)
        g=np.asarray(group,np.float32); extra.append(np.concatenate([g,g-g.mean(0),g-g.max(0)],1))
    e=embeddings.astype(np.float32); n=e/(np.linalg.norm(e,axis=2,keepdims=True)+1e-8)
    sim=np.einsum('nid,njd->nij',n,n)
    sim=np.sort(sim,axis=-1)[:,:,:3]
    return np.concatenate([base,np.asarray(extra),sim],2),valid,votes
