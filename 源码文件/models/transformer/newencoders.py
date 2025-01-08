
from tkinter import E
from tkinter.messagebox import NO
from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward,GateFFNLayer,RMSNorm
import torch
from torch import nn, norm
import torch.nn.functional as F
from models.transformer.attention import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from random import randrange
import torch.optim as optim
import torch.utils.data as Data
from torch import nn, einsum
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

def exists(val):
    return val is not None


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):  #如果有att-dim，dim—in 512 dim-out 1024 
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x,prev):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -&gt; b i j', q, k) * self.scale
        # prev1 = einsum('b i d, b j d -&gt; b i j', q, v) * self.scale
        if prev is not None:
            # prev=F.gelu(prev)
            sim =sim+prev
        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)
        
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -&gt; b i d', attn, v)
        
        return self.to_out(out),sim



class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()  #super().__init__()的作用也就显而易见了，就是执行父类的构造函数，使得我们能够调用父类的属性
        self.identity_map_reordering = identity_map_reordering

        # self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
        #                                 attention_module=attention_module,
        #                                 attention_module_kwargs=attention_module_kwargs)
        # self.mhatt =AFTLocal1(max_seqlen=50,dim=512)
        # self.attn = AFTLocal(max_seqlen=50,dim=512)
        self.attn = Attention(d_model, d_model, d_k, causal=True) if exists(d_k) else None
        self.mhatt  = EPM(d_model, 50, causal=True, act = nn.Identity(),heads=h, circulant_matrix = True)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
        # self.pwff = GateFFNLayer(d_model,dropout_rate=0.1)
    def forward(self, queries, keys, values,prev, attention_mask=None):
        # att = self.mhatt(queries, keys, values)
        # att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        gate_res,prev = self.attn(queries,prev) if exists(self.attn) else None

        att = self.mhatt(queries,gate_res = gate_res)

        ff = self.pwff(att)
        print(ff.shape)
        return ff,prev


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()  # (b_s, 1, 1, seq_len)
        prev =None
        outs = []
        out = input
        for l in self.layers:
            out,prev= l(out, out, out,prev,attention_mask)
            
        return out, attention_mask


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(MemoryAugmentedEncoder, self).forward(out)


class EPM(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        causal = False,
        act = nn.Identity(),
        heads = 1,
        init_eps = 1e-3,
        circulant_matrix = False
    ):
        super().__init__()
        dim_out = dim
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)

        self.act = act
        # self.line=nn.Linear(dim, dim_out)
        # parameters
         # self-atten 用的dim
        # sgu用的dim_ff = dim * ff_mult（4），即4倍
        # chunk之后，每个为2倍，用两倍的值进行attention
        if circulant_matrix:
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix
        shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, dim_seq))

    def forward(self, x, gate_res = None):
        device, n, h = x.device, x.shape[1], self.heads #n dim_sep(src_len)

        # res, gate = x.chunk(2, dim = -1) #分块 在dim_sep这个维度  res和gate 相当于伪代码中的u，v
        res = x
        gate = x
        gate = self.norm(gate)
        weight, bias = self.weight, self.bias

        if self.circulant_matrix:
            # build the circulant matrix

            dim_seq = weight.shape[-1]
            weight = F.pad(weight, (0, dim_seq), value = 0)
            weight = repeat(weight, '... n -&gt; ... (r n)', r = dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1):]

            # give circulant matrix absolute position awareness

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            weight = weight * rearrange(pos_x, 'h i -&gt; h i ()') * rearrange(pos_y, 'h j -&gt; h () j')

        if self.causal:
            weight, bias = weight[:, :n, :n], bias[:, :n]
            mask = torch.ones(weight.shape[-2:], device = device).triu_(1).bool()
            mask = rearrange(mask, 'i j -&gt; () i j')
            weight = weight.masked_fill(mask, 0.)

        gate = rearrange(gate, 'b n (h d) -&gt; b h n d', h = h)

        gate = einsum('b h n d, h m n -&gt; b h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -&gt; () h n ()')

        gate = rearrange(gate, 'b h n d -&gt; b n (h d)')
        print("2222222222")
        print(gate.shape)
        if exists(gate_res):
            gate = gate + gate_res
       
        return self.act(gate) * res


