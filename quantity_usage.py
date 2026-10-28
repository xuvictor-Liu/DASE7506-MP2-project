"""Quantity coverage in calculations versus restated givens; no answer references."""
import re
from collections import Counter
import numpy as np
from features_v2 import normalize
WORDS=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),range(21)))
WORDS.update(twice=2,thrice=3,dozen=12,half=.5,quarter=.25)
NUMBER=re.compile(r'(?<![\w.])\d+(?:\.\d+)?')
def number_words(text):
    return re.sub(r'\b('+ '|'.join(WORDS)+r')\b',lambda m:str(WORDS[m.group()]),text.lower())
def quantities(text):
    return [round(float(s),7) for s in NUMBER.findall(text.replace(',',''))]
def features(bank):
    output=[]
    for row in bank:
        q=number_words(row['question']); original=Counter(quantities(q)); distinct=set(original); groups=[]
        for candidate in row['candidates']:
            text=number_words(candidate['text']); all_values=set(quantities(text)); used=Counter(); first_used=Counter(); calc_lines=0; repeated=[]
            for line in text.splitlines():
                line=re.sub(r'^\s*(?:\d+[.)]|[-*])\s+','',line)
                if not re.search(r'=|\d\s*[+*/\u00d7\u00f7]|\\(?:times|div|cdot)|\d\s*-\s*\d',line): continue
                nums=quantities(normalize(line)); calc_lines+=1; used.update(nums)
                first_used.update(quantities(normalize(line.split('=')[0])))
                repeated.append(len(set(nums)&distinct))
            hits=distinct&set(used); lhs_hits=distinct&set(first_used); missing=distinct-hits; given_only=(distinct&all_values)-hits; total=sum(original.values()); count_covered=sum(min(original[k],used[k]) for k in original)
            group=[len(distinct),total,calc_lines,len(hits),len(lhs_hits),len(missing),len(given_only),len(hits)/max(1,len(distinct)),len(lhs_hits)/max(1,len(distinct)),count_covered/max(1,total),len(distinct-all_values),max(repeated,default=0),sum(repeated)/max(1,len(repeated)),len(set(used)-distinct),sum(used.values()),sum(k not in {0,1,2,100} for k in missing)]
            for word in ['each','every','total','remaining','left','more','less','twice','same','per']:
                group.extend([int(word in row['question'].lower())*len(missing),int(word in row['question'].lower())*len(given_only)])
            groups.append(group)
        g=np.asarray(groups,np.float32); output.append(np.concatenate([g,g-g.mean(0),g-g.max(0)],1))
    return np.asarray(output,np.float32)
