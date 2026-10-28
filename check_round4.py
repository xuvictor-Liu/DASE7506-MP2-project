"""Regression checks for the new parser and preservation of the frozen submission."""
import hashlib,json
from pathlib import Path
from trace_evidence import symbolic
from local_text import tokens

consistent=symbolic('x+108=160\nx=52')
assert len(consistent[1])==1 and abs(consistent[1][0]-52)<1e-6
assert not symbolic('x+108=160\nx=53')[1]
assert abs(symbolic('2x=10')[1][0]-5)<1e-6
assert not symbolic('nothing to solve')[1]
q,t=tokens('A worker makes 60 items in 2 hours.', '60 * 2 = 120\n#### 120')
assert 'quantity0' in q and 'quantity1' in q and 'finalvalue' in t
assert '60' not in t and '120' not in t
manifest=json.loads(Path('runs_v2/frozen_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('Round-four parser/token checks passed; all frozen V2 hashes unchanged.')
