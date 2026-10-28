"""Local consistency, answer mentions, and quantity provenance features."""
import ast,re
from fractions import Fraction
import numpy as np
from common import parse_answer
from features_v2 import normalize
from strict_arithmetic import extract
STOP=set('a an the of in on at to for by with and or is are was were be has have had how many much what total answer final therefore so we can find calculate step'.split())
def affine(node):
    if isinstance(node,ast.Constant) and type(node.value) in (int,float): return float(node.value),{}
    if isinstance(node,ast.Name) and re.fullmatch('[a-z]',node.id): return 0.,{node.id:1.}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        b,d=affine(node.operand); sign=-1 if isinstance(node.op,ast.USub) else 1; return b*sign,{k:v*sign for k,v in d.items()}
    if isinstance(node,ast.BinOp):
        a,ad=affine(node.left); b,bd=affine(node.right)
        if isinstance(node.op,(ast.Add,ast.Sub)):
            sign=-1 if isinstance(node.op,ast.Sub) else 1; return a+sign*b,{k:ad.get(k,0)+sign*bd.get(k,0) for k in ad.keys()|bd.keys()}
        if isinstance(node.op,ast.Mult):
            if ad and bd: raise ValueError()
            return a*b,({k:v*b for k,v in ad.items()} if ad else {k:v*a for k,v in bd.items()})
        if isinstance(node.op,ast.Div) and not bd and abs(b)>1e-12: return a/b,{k:v/b for k,v in ad.items()}
    raise ValueError()
def symbolic(text):
    t=normalize(text.lower()).replace('\\rightarrow','\n').replace('\\implies','\n').replace('\\\\','\n')
    t=re.sub(r'\b[a-z]{2,}\b','|',t); t=re.sub(r'(\d|\))[ \t]*([a-z])\b',r'\1*\2',t)
    constraints=[]
    for chunk in re.split(r'[|:\n]',t):
        chunk=re.sub(r'^\s*(?:\d+[.)]|[-*])\s+','',chunk)
        if '=' not in chunk: continue
        sides=chunk.split('=')
        for left,right in zip(sides,sides[1:]):
            try:
                a,ad=affine(ast.parse(left.strip(),mode='eval').body); b,bd=affine(ast.parse(right.strip(),mode='eval').body)
                d={k:ad.get(k,0)-bd.get(k,0) for k in ad.keys()|bd.keys()}
                if d and all(np.isfinite(v) for v in [a,b,*d.values()]): constraints.append((d,b-a))
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): pass
    names=sorted({k for d,b in constraints for k in d}); solved=[]
    if not names: return [0,0,0,0,0],solved
    a=np.array([[d.get(k,0) for k in names] for d,b in constraints]); b=np.array([b for d,b in constraints]); vals,_,rank,_=np.linalg.lstsq(a,b,rcond=None); residual=np.abs(a@vals-b)/(1+np.abs(b))
    if rank==len(names) and max(residual,default=0)<.001: solved=vals.tolist()
    return [len(names),len(constraints),int(rank),sum(residual>.01),min(max(residual,default=0),10)],solved
def features(bank):
    output=[]
    for row in bank:
        q=row['question'].lower(); qwords=set(re.findall('[a-z]+',re.split(r'[.!?]',q)[-2] if q.endswith('?') else q))-STOP; qnums=set(re.findall(r'\d+(?:\.\d+)?',q.replace(',','')))
        texts=[c['text'] for c in row['candidates']]; answers=[parse_answer(t) for t in texts]; values=[float(Fraction(a)) if a is not None else 0 for a in answers]; syms=[symbolic(t) for t in texts]; group=[]
        for i,t in enumerate(texts):
            normalized=normalize(t); eq=extract(t); nums=re.findall(r'\d+(?:\.\d+)?',normalized); tail=t[-400:].lower(); words=set(re.findall('[a-z]+',tail))-STOP
            boxes=re.findall(r'\\boxed\{([^{}]+)\}',t); boxanswers=[parse_answer('#### '+box) for box in boxes]; unique_boxes=set(a for a in boxanswers if a is not None)
            rhs=[b for a,b,err in eq]; lhs=[a for a,b,err in eq]; ans=values[i]
            overlap=len(qwords&words); lastnum=float(nums[-1]) if nums else 0
            f=syms[i][0]+[len(boxes),len(unique_boxes),len(unique_boxes)>1,len(qwords),overlap,overlap/max(1,len(qwords)),overlap/max(1,len(words)),len(set(nums)-qnums),lastnum==ans,
                sum(abs(a-ans)<1e-5*(1+abs(ans)) for a in rhs),sum(abs(a-ans)<1e-5*(1+abs(ans)) for a in lhs),abs(rhs[-1]-ans)/(1+abs(ans)) if rhs else 0,
                sum(abs(a-ans)<1e-5*(1+abs(ans)) for a in syms[i][1]),sum(any(abs(a-ans)<1e-5*(1+abs(ans)) for a in s[1]) for s in syms)]
            for w in ['wait','however','instead','actually','mistake','wrong','correct','not','round','approximately','profit','revenue','cost','remaining','original']:
                f.extend([tail.count(w),q.count(w)*tail.count(w)])
            group.append(f)
        g=np.array(group,np.float32); output.append(np.concatenate([g,g-g.mean(0),g-g.max(0)],1))
    return np.nan_to_num(np.array(output,np.float32),nan=0,posinf=100,neginf=-100)
