import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderUnit(nn.Module):
    def __init__(self, d_model=512,h=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model,h)#kdim=d_model//h,vdim=d_model//h)
        self.fc1  = nn.Linear(d_model, 4*d_model)
        self.fc2  = nn.Linear(4*d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model) 
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask):
        x0 = x
        print(x.shape)
        print(x.t(1).shape())
        x = x.t(1)
        x1 = self.mha(x,x,x,attn_mask=mask) # may need to define a container class that realizes masking
        x1 = self.ln1(x0 + self.drop(x1))
        x2 = self.fc2(F.relu(self.fc1(x1)))
        x2 = self.ln(x1 + self.drop(x2))
        return x2

class DecoderUnit(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model,h,kdim=d_model//h,vdim=d_model//h) # params are a hail mary
        self.mha  = nn.MultiheadAttention(d_model,h,kdim=d_model//h,vdim=d_model//h)
        self.fc1  = nn.Linear(d_model,4*d_model)
        self.fc2  = nn.Linear(4*d_model,d_model)
        self.ln1  = nn.LayerNorm(d_model) 
        self.ln2  = nn.LayerNorm(d_model) 
        self.ln3  = nn.LayerNorm(d_model) 
        self.drop = nn.Dropout(0.1)

    def forward(self, x, z, mask):
        x0 = x
        x1 = self.mha(x,x,x,attn_mask=mask)
        x1 = self.ln1(x0+self.drop(x1))
        x2 = sel.mha(z,z,x1,attn_mask)
        x2 = self.ln2(x1+self.drop(x2))
        x3 = self.fc2(F.relu(self.fc1(x2)))
        x3 = self.ln3(x2+self.drop(x3))
        return x3

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=512, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos/(1000**((2*i)/d_model)))
                pe[pos,i] = math.cos(pos/(1000**((2*(i+1)/d_model))))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe) # why this?
        self.drop = nn.Dropout(0.1)

    def forward(self,x):
        x = x*math.sqrt(self.d_model) # why this?
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len,:], requires_grad=False)#.cuda()  this looks sooo sus
        return x

if __name__ == "__main__":
    tf = Transformer()
    x = torch.zeros((1,28), dtype=torch.long)
    mask = torch.tensor([True]*28*28).reshape([1,28,28])
    print(mask.shape)
    print(tf(x,x,mask,mask))


