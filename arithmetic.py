"""Conservative arithmetic checks on explicit numeric equations, no eval."""
import ast
import operator
import re
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def value(s):
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (float,int): return float(n.value)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): return visit(n.operand)*(-1 if isinstance(n.op,ast.USub) else 1)
        raise ValueError()
    return visit(ast.parse(s.strip(),mode='eval').body)
def checks(text):
    t=text.replace(',','').replace('\\times','*').replace('\\cdot','*').replace('×','*').replace('÷','/').replace('\\div','/').replace('−','-')
    for _ in range(3): t=re.sub(r'\\(?:d?frac)\{([\d. +*/()\-]+)\}\{([\d. +*/()\-]+)\}',r'((\1)/(\2))',t)
    good=bad=0; errors=[]
    for m in re.finditer(r'(?<![\w.])([\d.() +*/\-]{3,})\s*=\s*([-+]?\d+(?:\.\d+)?)',t):
        lhs=m.group(1).strip()
        if not any(op in lhs for op in ['+','-','*','/']): continue
        try:
            a=value(lhs); b=float(m.group(2)); err=abs(a-b)/(1+abs(b))
            if err<1e-4: good+=1
            else: bad+=1
            errors.append(min(err,100))
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError): pass
    return [good,bad,good/(1+good+bad),sum(errors)/(1+len(errors)),max(errors,default=0)]
