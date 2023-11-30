import torch
import math
from torch import nn 
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 50304
    block_size: int = 1024
    n_embed:    int = 768
    n_head:     int = 12
    bias:       int = False
    dropout:  float = 0.0





class MultiheadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.Wq = nn.Linear(config.n_embed,config.n_embed)
        self.Wk = nn.Linear(config.n_embed,config.n_embed)
        self.Wv = nn.Linear(config.n_embed,config.n_embed)
        self.proj = nn.Linear(config.n_embed,config.n_embed,bias=config.bias)
        self.n_head = config.n_head
        self.drop = nn.Dropout(config.dropout)
        
    def forward(self,index):
        B,T,C = index.size()

        q = self.Wq(index)
        k = self.Wk(index)
        v = self.Wv(index)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        attention = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        ######### Apply masked attention
        attention = F.softmax(attention,dim=-1)
        attention = self.drop(attention)
        output = attention @ v 
        output = output.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        output = self.proj(output)
        output = self.drop(output)
        return output

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embed,config.n_embed*4)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(4*config.n_embed,config.n_embed)
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        y = self.drop(x)
        return y

class Decoder_Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed,bias=config.bias)
        self.attention = MultiheadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed,bias=config.bias)
        self.MLP = MLP(config)
    
    def forward(self,x):
        x = x+self.attention(self.ln1(x))
        y = x+self.MLP(self.ln2(x))
        return y

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.Token_embeddings = nn.Embedding(config.vocab_size,config.n_embed)
        self.Positional_embeddings = nn.Embedding(config.block_size,config.n_embed)
        self.Decoder = nn.ModuleList([Decoder_Block(config) for _ in range(config.n_head)])
        self.dropout = nn.Dropout(config.dropout)
        self.lnorm = nn.LayerNorm(config.n_embed,bias=config.bias)
        self.LMhead = nn.Linear(config.n_embed,config.vocab_size,bias=config.bias)
    
    def forward(self,index,targets=None):
        device = index.device
        b,t = index.size()
        positions = torch.arange(0,t, dtype=torch.long,device=device)

        token_embeddings = self.Token_embeddings(index)
        positional_embeddings = self.Positional_embeddings(positions)
        x = self.dropout(token_embeddings + positional_embeddings)
        for decoder_block in self.Decoder:
            x = decoder_block(x)
        y = self.lnorm(x)

        if targets is None:
            logits = self.LMhead(x[:,[-1],:])
            loss = None
        else:
            logits = self.LMhead(x)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1), ignore_index=-1)

        return logits,loss
