import torch
import torch.nn as nn
import numpy as np

class MutiHeadAttention(nn.Module):
    def __init__(self,d_k,d_v,d_model,num_heads,p=0.):
        super(MutiHeadAttention,self).__init__()
        self.d_model = d_model # 整体向量维度-512
        self.d_k = d_k # 每个头特征维度 512/8=64
        self.d_v = d_v
        self.num_heads = num_heads # 头数-8
        self.dropout = nn.Dropout(p)

        self.W_Q = nn.Linear(d_model,d_k * num_heads)
        self.W_K = nn.Linear(d_model,d_k * num_heads)
        self.W_V = nn.Linear(d_model,d_k * num_heads)
        self.W_out = nn.Linear(d_k * num_heads,d_model)

        nn.init.normal_(self.W_Q.weight,mean=0,std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight,mean=0,std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight,mean=0,std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight,mean=0,std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self,Q,K,V,attn_mask,**kwargs):
        N = Q.size(0)
        q_len,k_len = Q.size(1),K.size(1)
        d_k,d_v = self.d_k,self.d_v
        num_heads = self.num_heads

        Q = self.W_Q(Q).view(N,-1,num_heads,d_k).transpose(1,2)
        K = self.W_K(K).view(N,-1,num_heads,d_k).transpose(1,2)
        V = self.W_V(V).view(N,-1,num_heads,d_v).transpose(1,2)

        scores = torch.matual(Q,K.transpose(-1,-2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask,-1e4)
        attns = torch.softmax(scores,dim=-1)
        attns = self.dropout(attns)

        output = torch.matmul(attns,V)

        output = output.transpose(1,2).contiguous().reshape(N,-i,d_v*num_heads)
        output = self.W_out(output)

        return output

def pos_sinusoid_embedding(seq_len,d_model):
    embeddings = torch.zeros((seq_len,d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:,i] = f(torch.arange(0,seq_len) / np.power(1e4,2*(i//2) / d_model))
        return embeddings.float()

class PoswiseFFN(nn.Module):
    def __init__(self,d_model,d_ff,p=0.):
        super(PoswiseFFN,self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model,d_ff,1,1,0)
        self.conv2 = nn.Conv1d(d_ff,d_model,1,1,0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self,x):
        out = self.conv1(X.transpose(1,2))
        out = self.relu(out)
        out = self.conv2(out).transpose(1,2)
        out = self.dropout(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self,dim,n,dff,dropout_posffn,dropout_attn):
        assert dim % n == 0
        hdim = dim // n
        super(EncoderLayer,self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MutiHeadAttention(hdim,hdim,dim,n,dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim,dff,p=dropout_posffn)

    def forward(self,enc_in,attn_mask):
        residual = enc_in
        context = self.multi_head_attn(enc_in,enc_in,enc_in,attn_mask)
        out = self.norm1(residual + context)
        residual = out
        out = self.poswise_ffn(out)
        out = self.norm2(residual + out)
        return out

class Encoder(nn.Module):
    def __init__(
            self,dropout_emb,dropout_posffn,dropout_attn,
            num_layers,enc_dim,num_heads,dff,tgt_len,
    ):
        super(EncoderLayer,self).__init__()
        self.tgt_len = tgt_len
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len,enc_dim),freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim,num_heads,dff,dropout_posffn,dropout_attn) for _ in range(num_layers)]
        )

    def forward(self,X,X_lens,mask=None):
        batch_size,seq_len,d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len,device=X.device))
        out = self.emb_dropout(out)
        for layer in self.layers:
            out = layer(out,mask)
        return out