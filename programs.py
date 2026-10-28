"""Learnable arithmetic templates extracted only from supplied training solutions."""
import ast,re,operator
from fractions import Fraction
import numpy as np
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
SYMS={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}
CONSTANTS={0,1,2,3,4,5,7,10,12,24,60,100,1000}
PATTERN=r'(?<![\w.])\d+(?:\.\d+)?|(?<![\w.])\.\d+'
def question(q):
    q=q.lower().replace(',',''); nums=[]
    def sub(m):
        nums.append(float(m.group())); return ' number'+str(len(nums)-1)+' '
    text=re.sub(PATTERN,sub,q)
    return text,nums
def evaluate(program,nums):
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return float(n.value)
        if isinstance(n,ast.Name) and re.fullmatch('n[0-9]+',n.id): return nums[int(n.id[1:])]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): return visit(n.operand)*(-1 if isinstance(n.op,ast.USub) else 1)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.left),visit(n.right))
        raise ValueError()
    try:
        answer=visit(program if isinstance(program,ast.AST) else ast.parse(program,mode='eval').body)
        return answer if np.isfinite(answer) and abs(answer)<1e12 else None
    except (ValueError,TypeError,IndexError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): return None
def extract(q,answer):
    _,nums=question(q); known={}
    for i,n in enumerate(nums): known.setdefault(round(n,8),'n'+str(i))
    def replace(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=float(n.value); key=round(v,8)
            if key in known: return known[key]
            if v in CONSTANTS: return str(int(v))
            # Common decimal proportions can be derived from an explicit percentage.
            if round(v*100,8) in known: return '('+known[round(v*100,8)]+'/100)'
            raise ValueError()
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): return ('-' if isinstance(n.op,ast.USub) else '+')+replace(n.operand)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return '('+replace(n.left)+SYMS[type(n.op)]+replace(n.right)+')'
        raise ValueError()
    for equation in re.findall(r'<<([^<>]+)>>',answer):
        if '=' not in equation: continue
        left,right=equation.rsplit('=',1)
        try:
            value=float(Fraction(right.replace(',','').strip())); program=replace(ast.parse(left.replace(',',''),mode='eval').body)
            pred=evaluate(program,nums)
            if pred is not None and abs(pred-value)<1e-5*(1+abs(value)) and len(program)<=180:
                known.setdefault(round(value,8),program)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): pass
    from common import parse_answer
    gold=parse_answer(answer)
    if gold is None: return None
    p=known.get(round(float(Fraction(gold)),8))
    if p and 'n' in p and any(op in p for op in '+-*/'): return p
    return None

def program_support(bank,b):
    from common import parse_answer
    textnums=[question(r['question']) for r in bank]
    probs=b['model'].predict_proba(b['vectorizer'].transform([t for t,n in textnums]))
    programs=[ast.parse(p,mode='eval').body for p in b['model'].classes_]
    support=np.zeros((len(bank),4)); confidence=probs.max(1)
    for i,(row,(_,nums)) in enumerate(zip(bank,textnums)):
        answers=[parse_answer(c['text']) for c in row['candidates']]
        vals=[float(Fraction(a)) if a is not None else None for a in answers]
        for j,p in enumerate(programs):
            value=evaluate(p,nums)
            if value is None: continue
            for k,a in enumerate(vals):
                if a is not None and abs(a-value)<1e-5*(1+abs(a)): support[i,k]+=probs[i,j]
    return support,confidence
