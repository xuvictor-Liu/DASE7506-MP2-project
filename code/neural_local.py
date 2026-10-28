"""Small randomly initialized text CNN and permutation-equivariant set attention."""
import torch
from torch import nn
class LocalVerifier(nn.Module):
    def __init__(self,input_dim,vocab_size,kind='cnn',dropout=.25):
        super().__init__(); self.kind=kind
        self.numeric=nn.Sequential(nn.Linear(input_dim,128),nn.GELU(),nn.Dropout(dropout))
        if kind=='cnn':
            self.embedding=nn.Embedding(vocab_size,24,padding_idx=0)
            self.convs=nn.ModuleList([nn.Conv1d(24,24,k,padding=k//2) for k in [3,5,7]])
            self.qproj=nn.Linear(24,72)
            self.head=nn.Sequential(nn.Linear(128+72+72+72,96),nn.GELU(),nn.Dropout(dropout),nn.Linear(96,1))
        else:
            layer=nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=256,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
            self.layers=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
            self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,1))
    def forward(self,x,text,q,valid):
        batch=x.shape[0]; h=self.numeric(x)
        if self.kind=='cnn':
            t=text.reshape(-1,text.shape[-1]); qt=q.reshape(-1,q.shape[-1]); emb=self.embedding(t).transpose(1,2); mask=t.ne(0)[:,None,:]
            maxima=[]; means=[]
            for conv in self.convs:
                z=torch.relu(conv(emb)); maxima.append(z.masked_fill(~mask,-1e4).amax(2)); means.append((z*mask).sum(2)/mask.sum(2).clamp_min(1))
            pool=torch.cat(maxima,1); avg=torch.cat(means,1)
            qe=self.embedding(qt); qm=qt.ne(0)[:,:,None]; qe=(qe*qm).sum(1)/qm.sum(1).clamp_min(1); interaction=pool*torch.tanh(self.qproj(qe))
            return self.head(torch.cat([h.reshape(-1,128),pool,avg,interaction],1)).reshape(batch,4)
        return self.head(self.layers(h,src_key_padding_mask=~valid)).squeeze(-1)
