import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class Transformer(nn.Module):
    def __init__(self, vocab_size=666, d_model=512):
        super().__init__()
        # Input embedding
        self.inemb = nn.Embedding(vocab_size,d_model)
        # Input Positional Encoding
        self.inpos = PositionalEncoder()
        # Input Encoding
        self.enc1 = EncoderUnit()
        self.enc2 = EncoderUnit()
        self.enc3 = EncoderUnit()
        self.enc4 = EncoderUnit()
        self.enc5 = EncoderUnit()
        self.enc6 = EncoderUnit()
        # Output Embedding
        self.outemb = nn.Embedding(vocab_size,d_model)
        # Output Positional Encoding
        self.outpos = PositionalEncoder()
        self.dec2 = DecoderUnit()
        self.dec3 = DecoderUnit()
        self.dec4 = DecoderUnit()
        self.dec5 = DecoderUnit()
        self.dec6 = DecoderUnit()
        # De-Embedding
        self.fc = nn.Linear(d_model,vocab_size)
        # Non parametric layers
        self.drop = nn.Dropout(0.1)

    def forward(self,x,y,input_mask, output_mask):
        # Input path
        x = self.inemb(x)
        x = self.inpos(x)
        x = self.enc1(x,input_mask)
        x = self.enc2(x,input_mask)
        x = self.enc3(x,input_mask)
        x = self.enc4(x,input_mask)
        x = self.enc5(x,input_mask)
        z = self.enc6(x,input_mask)
        # Output path
        y = self.outemb(y)
        y = self.outpos(y)
        y = self.dec1(y,z,out_mask)
        y = self.dec2(y,z,out_mask)
        y = self.dec3(y,z,out_mask)
        y = self.dec4(y,z,out_mask)
        y = self.dec5(y,z,out_mask)
        y = self.dec6(y,z,out_mask)
        # Result
        y = self.fc(y)
        return y

if __name__ == "__main__":
    tf = Transformer()
    x = torch.zeros((1,28), dtype=torch.long)
    mask = torch.tensor([True]*28*28).reshape([1,28,28])
    print(mask.shape)
    print(tf(x,x,mask,mask))


