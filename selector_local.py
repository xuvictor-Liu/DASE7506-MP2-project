"""Offline CPU inference for experimental randomly initialized local verifiers."""
import numpy as np
import torch
from neural_local import LocalVerifier
from local_text import encode
from features_v2 import features_v2
from strict_arithmetic import features as arithmetic_features
def predict(bank,arrays,checkpoint):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=False)
    e=arrays['embeddings']; f,valid,_=features_v2(bank,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),f,arithmetic_features(bank)],2)
    x=np.clip((raw-model['mean'])/model['scale'],-10,10).astype(np.float32); t,q=encode(bank,model['vocab']); cfg=model['config']; net=LocalVerifier(x.shape[-1],cfg['vocab_size'],cfg['kind'],cfg['dropout']); scores=np.zeros((len(x),4))
    with torch.no_grad():
        for state in model['states']:
            net.load_state_dict(state); net.eval()
            for start in range(0,len(x),32):
                args=[torch.as_tensor(a[start:start+32]) for a in [x,t,q,valid]]
                scores[start:start+32]+=net(*args).numpy()/len(model['states'])
    return np.where(valid,scores,-1e20).argmax(1).tolist()
