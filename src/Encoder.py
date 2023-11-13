import copy,math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import gelu
PAD_NUM = 0

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query,key,value,mask=None,dropout=None):
    """
    :param query: (batch_size,h,seq_len,embedding)
    :param key:
    :param value:
    :param mask: (batch_size,1,1,seq_len)
    :param dropout:
    :return: (batch_size,h,seq_len,embedding)
    """
    d_k = query.size(-1)
    score = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == PAD_NUM,-1e9)
    p_atten = F.softmax(score,dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten,value),p_atten


class AtomPositionalEncoding(nn.Module):
    def __init__(self, dropout, pad_num, d_model):
        super(AtomPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.pad_num = pad_num
    
    def get_dist_matrix(self, num_mol, max_num_atom, coords_data, lattice):
        #num_mol = x.size(0) #batch_size
        #max_num_atom = x.size(1)  #max_seq_len
        temp_padded_coords = np.zeros((num_mol, 27, max_num_atom, 3))
        lattice_reshape = lattice.reshape(-1, 3, 3)
        for i in range(num_mol):
            padded_mol = np.pad(coords_data[i], ((0, max_num_atom - coords_data[i].shape[0]), (0, 0)), 'constant',
                                constant_values=(self.pad_num, self.pad_num))
            lattice_a = lattice_reshape[i,0,:]
            lattice_b = lattice_reshape[i,1,:]
            lattice_c = lattice_reshape[i,2,:]
            count = 0
            #temp_padded_coords[i] = torch.from_numpy(padded_mol)
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    for l in [0, -1, 1]:
                        tran_vec = j*lattice_a+k*lattice_b+l*lattice_c
                        #print(padded_mol,tran_vec)
                        temp_padded_coords[i][count] = padded_mol + tran_vec
                        count += 1
        return temp_padded_coords
        #return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords.cuda()))))
        #return temp_padded_coords

    def forward(self, x, padded_coords, central_atom_index):
        temp_padded_coords = torch.tensor(padded_coords - padded_coords[:, 0, central_atom_index, :].reshape(padded_coords.shape[0],1,1,padded_coords.shape[3]),dtype=torch.float32)
        return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords.cuda()))))
        ###########################cuda()#############################
        #return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords))))      


class UnitPositionalEncoding(nn.Module):
    def __init__(self, dropout, pad_num, d_model):
        super(UnitPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.pad_num = pad_num
    
    def get_dist_matrix(self, num_mol, lattice):
        #num_mol = x.size(0) #batch_size
        #max_num_atom = x.size(1)  #max_seq_len
        temp_padded_coords = np.zeros((num_mol, 27, 3)) #3维坐标点, 27 个邻接unit
        lattice_reshape = lattice.reshape(-1,3,3)
        for i in range(num_mol):
            lattice_a = lattice_reshape[i,0,:]
            lattice_b = lattice_reshape[i,1,:]
            lattice_c = lattice_reshape[i,2,:]
            count = 0
            #padded_mol = torch.from_numpy(padded_mol)
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    for l in [0, -1, 1]:
                        tran_vec = j*lattice_a+k*lattice_b+l*lattice_c
                        #print(padded_mol.shape,tran_vec.shape)
                        temp_padded_coords[i][count] = tran_vec
                        count += 1
        return temp_padded_coords

    def forward(self, x, padded_coords, central_atom_index):
        temp_padded_coords = torch.tensor(padded_coords - padded_coords[:, central_atom_index, :].reshape(padded_coords.shape[0],1,padded_coords.shape[2]),dtype=torch.float32)
        ###########################cuda()#############################
        return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords.cuda()))))
        #return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords))))


class UnitEncoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout=0.1): # layer = EncoderLayer 
        super(UnitEncoder, self).__init__()
        self.layers = clone(layer,N)
        self.norm = nn.LayerNorm(layer.layerNormSize)
        self.atom_position_encoding = AtomPositionalEncoding(dropout, PAD_NUM, d_model)

    def forward(self, x, mask, coords_data, lattice):
        num_mol = x.size(0) #batch_size
        max_num_atom = x.size(1)  #max_seq_len
        padded_coords_matrix = self.atom_position_encoding.get_dist_matrix(num_mol, max_num_atom, coords_data, lattice)

        for layer in self.layers:
            x = layer(x, mask, padded_coords_matrix, self.atom_position_encoding)
        return self.norm(x)


class CrystalEncoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout=0.1):
        super(CrystalEncoder, self).__init__()
        self.layers = clone(layer,N)
        self.norm = nn.LayerNorm(layer.layerNormSize)
        self.unit_position_encoding = UnitPositionalEncoding(dropout, PAD_NUM, d_model)
    
    def forward(self, x, lattice):
        num_mol = x.size(0)
        padded_coords_matrix = self.unit_position_encoding.get_dist_matrix(num_mol, lattice)

        for layer in self.layers:
            x = layer(x, padded_coords_matrix, self.unit_position_encoding)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, layerNormSize, p):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(layerNormSize)
        self.dropout = nn.Dropout(p)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class CrystalEncoderLayer(nn.Module):
    def __init__(self,layerNormSize,self_atten,feed_forward,dropout):
        # self_atten = MultiHeadAttention, feed_forward = PositionwiseFeedForward
        super(CrystalEncoderLayer, self).__init__()
        self.self_atten = self_atten
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(layerNormSize,dropout),2)
        self.layerNormSize = layerNormSize

    def forward(self,x, padded_coords_matrix, position_encodeing):
        # num_mol * 27 * max_num_atom * emdding_dim
        # mask : num_mol * max_num_atom

        #############################cuda()############################################
        mask_pre_use = torch.zeros(27).cuda()
        #mask_pre_use = torch.zeros(27)
        mask_pre_use[0] = 1
        atom_reps = position_encodeing(x.unsqueeze(1), padded_coords_matrix, 0)
        atom_reps_reshape = atom_reps.reshape(atom_reps.shape[0],-1,atom_reps.shape[-1])
        atom_reps_res_temp = self.sublayer[0](atom_reps_reshape,lambda atom_reps_reshape:self.self_atten(atom_reps_reshape,atom_reps_reshape,atom_reps_reshape))
        atom_reps_res = mask_pre_use.unsqueeze(-1)*(atom_reps_res_temp.reshape(atom_reps.shape[0], 27, atom_reps.shape[2]))
        return self.sublayer[1](atom_reps_res.sum(1),self.feed_forward)


class AtomEncoderLayer(nn.Module):
    def __init__(self,layerNormSize,self_atten,feed_forward,dropout):
        # self_atten = MultiHeadAttention, feed_forward = PositionwiseFeedForward
        super(AtomEncoderLayer, self).__init__()
        self.self_atten = self_atten
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(layerNormSize,dropout),2)
        self.layerNormSize = layerNormSize

    def forward(self,x,mask,padded_coords_matrix, position_encodeing):
        # num_mol * 27 * max_num_atom * emdding_dim
        # mask : num_mol * max_num_atom
        #print(mask.shape)
        mask_enhance = mask.repeat(1,27,1).reshape(mask.shape[0],-1).unsqueeze(1)
        '''
        mask_pre = torch.zeros(27, x.shape[1], x.shape[2]).cuda()
        mask_pre_use = mask_pre
        mask_pre_use[0,0,:] = 1
        '''
        #############################cuda()############################################
        mask_pre_use = torch.zeros(27, x.shape[1], x.shape[2]).cuda()
        #mask_pre_use = torch.zeros(27, x.shape[1])
        mask_pre_use[0,0] = 1
        atom_reps = position_encodeing(x.unsqueeze(1), padded_coords_matrix, 0)
        atom_reps_reshape = atom_reps.reshape(atom_reps.shape[0],-1,atom_reps.shape[-1])
        atom_reps_res_temp = self.sublayer[0](atom_reps_reshape,lambda atom_reps_reshape:self.self_atten(atom_reps_reshape,atom_reps_reshape,atom_reps_reshape, mask_enhance))
        atom_reps_res = mask_pre_use*(atom_reps_res_temp.reshape(atom_reps.shape[0], 27, atom_reps.shape[2], -1))

        for i in range(1,x.size(1),1):
            '''
            mask_pre_use = mask_pre
            mask_pre_use[0,i,:] = 1
            '''
            #############################cuda()############################################
            mask_pre_use = torch.zeros(27, x.shape[1], x.shape[2]).cuda()
            #mask_pre_use = torch.zeros(27, x.shape[1])
            mask_pre_use[0,i] = 1            
            atom_reps = position_encodeing(x.unsqueeze(1), padded_coords_matrix, i)
            atom_reps_reshape = atom_reps.reshape(atom_reps.shape[0],-1,atom_reps.shape[-1])
            atom_reps_res_temp = self.sublayer[0](atom_reps_reshape,lambda atom_reps_reshape:self.self_atten(atom_reps_reshape,atom_reps_reshape,atom_reps_reshape, mask_enhance))
            atom_reps_res = atom_reps_res + mask_pre_use*atom_reps_res_temp.reshape(atom_reps.shape[0], 27, atom_reps.shape[2], -1)
            # atom_reps_res: num_mol*27*max_num_atom*embedding

        return self.sublayer[1](atom_reps_res.sum(1),self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model
        self.atten = None
        self.dropout = nn.Dropout(dropout)
        self.linears = clone(nn.Linear(d_model,d_model),4)

    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)
        query,key,value = [l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size,1,dk) => (batch_size,1,1,seq_len)
        x,self.atten = attention(query,key,value,mask,self.dropout)
        return self.linears[-1](x.transpose(1,2).contiguous().view(batch_size,-1,self.d_model))

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


if __name__ == "__main__":
    embedding = Embedding(50,512)
    test_data = embedding(torch.randint(0,50,(2,5)))
    mask = torch.zeros(2,1,5)
    multi = MultiHeadAttention(8,512)
    encoderlayer = EncoderLayer(512,multi,PositionwiseFeedForward(512,256),0.1)
    res = encoderlayer(test_data,mask)
    print(res.shape)

