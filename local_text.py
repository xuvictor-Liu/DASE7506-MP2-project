"""Training-only vocabulary and quantity-anchored tokenization, no pretrained tokenizer."""
import re
from fractions import Fraction
from collections import Counter
import numpy as np
from common import parse_answer
TOKEN=re.compile(r'\d+(?:\.\d+)?|[a-z]+|[+*/=<>%()-]')
def tokens(question,text):
    qnums=re.findall(r'\d+(?:\.\d+)?',question.replace(',',''))
    answer=parse_answer(text)
    def convert(t):
        low=t.lower().replace(',','').replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
        out=[]
        for token in TOKEN.findall(low):
            if token[0].isdigit():
                if answer is not None and str(Fraction(token))==answer: token='finalvalue'
                elif token in qnums: token='quantity'+str(min(qnums.index(token),9))
                else: token='derivedvalue'
            out.append(token)
        return out
    return convert(question),convert(text)
def vocabulary(bank,size=6000):
    count=Counter()
    for row in bank:
        for c in row['candidates']:
            q,t=tokens(row['question'],c['text']); count.update(q); count.update(t)
    return {word:i+2 for i,(word,n) in enumerate(count.most_common(size-2)) if n>=2}
def encode(bank,vocab,length=288,qlength=64):
    texts=np.zeros((len(bank),4,length),np.int64); questions=np.zeros((len(bank),4,qlength),np.int64)
    for i,row in enumerate(bank):
        for j,c in enumerate(row['candidates']):
            q,t=tokens(row['question'],c['text'])
            if len(t)>length: t=t[:64]+t[-(length-64):]
            q=q[-qlength:]
            texts[i,j,:len(t)]=[vocab.get(w,1) for w in t]; questions[i,j,:len(q)]=[vocab.get(w,1) for w in q]
    return texts,questions
