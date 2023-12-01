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
    n_layer:    int = 12
    batch_size: int = 32
    EPOCH:      int = 2000
    learning_rate:int= 5e-5

def create_mask(T):
        mask = torch.full([T,T],float("-inf"))
        mask = torch.triu(mask,diagonal=1)
        return mask



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
        ##Masking
        attention = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        attention += create_mask(T)
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

class GPT_model(nn.Module):
    def __init__(self,config=Config):
        super().__init__()
        self.Token_embeddings = nn.Embedding(config.vocab_size,config.n_embed)
        self.Positional_embeddings = nn.Embedding(config.block_size,config.n_embed)
        self.Decoder = nn.ModuleList([Decoder_Block(config) for _ in range(config.n_layer)])
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
    def generate(self, idx, max_new_tokens,blocksize): 
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -blocksize:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx     
class GPT:
    def __init__(self,
                 Data,
                 vocab_size,
                 block_size,
                 n_embed,
                 n_head,
                 bias,
                 dropout,
                 n_layer,
                 batch_size,
                 EPOCH,
                 learning_rate):
        self.config = Config(vocab_size=vocab_size,
                             block_size=block_size,
                             n_embed=n_embed,
                             n_head=n_head,
                             bias=bias,
                             dropout=dropout,
                             n_layer=n_layer,
                             batch_size=batch_size,
                             EPOCH=EPOCH,
                             learning_rate=learning_rate)
        self.gpt = GPT_model(self.config)
        self.Data = Data
        self.words = sorted([i for i in set(Data)])
        idxtos = {idx:w for idx,w in enumerate(self.words)}
        stoidx = {w:idx for idx,w in enumerate(self.words)}
        self.encode = lambda s: [stoidx[i] for i in s]
        self.decode = lambda i: "".join(idxtos[j] for j in i)
        self.data_tensor = torch.tensor(self.encode(Data), dtype=torch.long)
        n = int(0.9*len(self.data_tensor)) # first 90% will be train, rest val
        self.train_data = self.data_tensor[:n]
        self.val_data = self.data_tensor[n:]
        self.eval_iters = self.config.EPOCH // 10
        self.optimizer = torch.optim.AdamW(self.gpt.parameters(), lr=self.config.learning_rate)
        
    
    def get_batch(self,split,batch_size):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(self.data_tensor) - self.config.block_size, (batch_size,))
        x = torch.stack([self.data_tensor[i:i+self.config.block_size] for i in ix])
        y = torch.stack([self.data_tensor[i+1:i+self.config.block_size+1] for i in ix])
        return x, y
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.gpt.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split,self.config.batch_size)
                logits, loss = self.gpt(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.gpt.train()
        return out
    
    def train(self):
        for iter in range(self.config.EPOCH):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_iters == 0:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.get_batch('train',self.config.batch_size)

            # evaluate the loss
            logits, loss = self.gpt(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
    
    def generate(self,max_new_tokens):
        context = torch.zeros((1, 1), dtype=torch.long)
        print(self.decode(self.gpt.generate(context, max_new_tokens=max_new_tokens,blocksize=self.config.block_size)[0].tolist()))


        

        
        





