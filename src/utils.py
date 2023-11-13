from turtle import forward
import numpy as np
import copy,math
import torch
import torch.nn as nn
import torch.nn.functional as F
PAD_NUM = 0
N_DIM = 3
coords_train_set,coords_eval_set,coords_test_set = None,None,None


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def pad_to_same(data, pad_num):
    max_len = 0

    for i in range(len(data)):
        if max_len < len(data[i]):
            max_len = len(data[i])
    new_data = np.zeros((len(data), max_len))
    for i in range(len(data)):
        new_data[i] = np.pad(data[i], (0, (max_len - len(data[i]))), 'constant', constant_values=(pad_num, pad_num))

    return torch.from_numpy(new_data)


class BatchSampler(object):

    def __init__(self, batch_size, dataset, drop_last=False): #dataset: the full train/val/test datasets
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_data = len(dataset)
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.random.permutation(self.num_data)
        batch = []
        for i in indices:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_data // self.batch_size
        else:
            return (self.num_data + self.batch_size - 1) // self.batch_size


class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)  


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, pad_num, d_model):
        super(PositionalEncoding).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.pad_num = pad_num
    
    def get_dist_matrix(self, num_mol, max_num_atom, coords_data, lattice):
        #num_mol = x.size(0) #batch_size
        #max_num_atom = x.size(1)  #max_seq_len
        temp_padded_coords = torch.zeros(num_mol, 27, max_num_atom, 3) #3维坐标点, 27 个邻接unit
        for i in range(num_mol):
            padded_mol = np.pad(coords_data[i], ((0, max_num_atom - coords_data[i].shape[0]), (0, 0)), 'constant',
                                constant_values=(self.pad_num, self.pad_num))
            lattice_reshape = lattice.reshape(-1,3,3)
            lattice_a = lattice_reshape[0]
            lattice_b = lattice_reshape[1]
            lattice_c = lattice_reshape[2]
            count = 0
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    for l in [0, -1, 1]:
                        temp_padded_coords[i][count] = torch.from_numpy(padded_mol+j*lattice_a+k*lattice_b+l*lattice_c)
                        count += 1
        return temp_padded_coords


    def forward(self, x, padded_coords, central_atom_index):
        temp_padded_coords = padded_coords - padded_coords[:, 0, central_atom_index, :].unsqueeze(1).unsqueeze(1)
        ###################################cuda()##############################################
        return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords.cuda()))))
        #return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords))))

'''
class PositionalEncoding(nn.Module):
    def __init__(self, dropout,d_model,pad_num):  # 修改：(self,  coords, dropout)
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model,d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.pad_num = pad_num
        
    def forward(self,x,coords_data):
        # x: batch_size,max_seq_len,d_model
#         print("hello",coords_data.shape)
        num_mol = x.size(0) #batch_size  
        max_num_atom = x.size(1)  #max_seq_len
        temp_padded_coords = torch.zeros(num_mol, max_num_atom, self.d_model)
        for i in range(num_mol):
            padded_mol = np.pad(coords_data[i], ((0, max_num_atom - coords_data[i].shape[0]), (0, 0)), 'constant',
                                constant_values=(self.pad_num, self.pad_num))
            temp_padded_coords[i] = torch.from_numpy(padded_mol)
        return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords.cuda()))))
'''


class Set2Set(nn.Module):
    def __init__(self,atom_feature_size,T = 10):
        super(Set2Set, self).__init__()
        self.input_size =  atom_feature_size*2
        self.hidden_size = atom_feature_size   # in LSTMCell: hidden_size == atom_feature_size
        self.LSTMCell = nn.LSTMCell(self.input_size,self.hidden_size)
        self.T = T # T : the num of operations in LSTMCell
    def forward(self,x,mask_for_set2set): # x : atom_feature (batch_size,seq_len,atom_feature_size)  atom_feature_size == hidden_size
#         print(x.dtype)
#         x = torch.tensor(x, dtype=torch.float32).cuda()
#         print(x.dtype)
        batch_size = x.shape[0]
        ##############################cuda()#####################################
        mask_for_set2set = mask_for_set2set.unsqueeze(dim = 2).cuda()
        #mask_for_set2set = mask_for_set2set.unsqueeze(dim = 2)
        
        ############################cuda()#####################################
        h,c = torch.zeros(batch_size,self.hidden_size).cuda(),torch.zeros(batch_size,self.hidden_size).cuda()  #(batch_size,hidden_size)
        #h,c = torch.zeros(batch_size,self.hidden_size),torch.zeros(batch_size,self.hidden_size) #(batch_size,hidden_size)
        ##############################cuda()#################################
        q_star = torch.zeros(batch_size,self.input_size).cuda()
        #q_star = torch.zeros(batch_size,self.input_size)
#         print(h.dtype,c.dtype,q_star.dtype)
        for i in range(self.T):
#             print(q_star.cuda().dtype,h.cuda().dtype,c.cuda().dtype)
            h,c = self.LSTMCell(q_star,(h,c)) # h,c: (batch_size,hidden_size)
            query_vector = torch.unsqueeze(h,dim = 2) # (batch_size,hidden_size,1)
            #(batch_size,seq_len,atom_feature_size)* (batch_size,hidden_size,1)  hidden_size == atom_feature_size
            e_vector = torch.matmul(x,query_vector) #(batch_size,seq_len,1)
            e_vector = e_vector.masked_fill(mask_for_set2set, -1e9)
            a_vector = F.softmax(e_vector, dim=1) #softmax  (batch_size,seq_len,1)
            a_vector = a_vector.permute(0,2,1) #(batch_size,1,seq_len)
            # (batch_size,1,seq_len) * (batch_size,seq_len,atom_feature_size)
            r_vector = torch.matmul(a_vector,x)#(batch_size,1,atom_feature_size)
            r_vector = torch.squeeze(r_vector,dim = 1) # (batch_size,atom_feature_size)
            q_star = torch.cat([h,r_vector], dim = 1) #(batch_size,input_size)  input_size == 2 *  hidden_size == 2 * atom_feature_size
        return q_star  #(batch,atom_feature_size*2)

class Regression_fun_without_lattice(nn.Module):
    def __init__(self, output_size):
        super(Regression_fun_without_lattice, self).__init__()
#         self.fc1 = nn.Linear(output_size+9, int(output_size*1.5))  #???? 512+9?
#         self.feature = nn.Linear(9,output_size)
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4),2)
        
    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
#         x = x + self.feature(lattice)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))


class Regression_fun(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4), 1)
    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))


class Regression_fun_with_lattice(nn.Module):
    def __init__(self, output_size):
        super(Regression_fun_with_lattice, self).__init__()
#         self.fc1 = nn.Linear(output_size+9, int(output_size*1.5))  #???? 512+9?
        self.feature = nn.Linear(9,output_size)
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4),2)
        
    def forward(self, x,lattice):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        x = x + self.feature(lattice)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))
    
class energy_without_lattice(nn.Module):
    def __init__(self, output_size):
        super(energy_without_lattice, self).__init__()
#         self.fc1 = nn.Linear(output_size+9, int(output_size*1.5))  #???? 512+9?
#         self.feature = nn.Linear(9,output_size)
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4),1)
        
    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
#         x = x + self.feature(lattice)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))

class energy_with_lattice(nn.Module):
    def __init__(self, output_size):
        super(energy_with_lattice, self).__init__()
#         self.fc1 = nn.Linear(output_size+9, int(output_size*1.5))  #???? 512+9?
        self.feature = nn.Linear(9,output_size)
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4),1)
        
    def forward(self, x,lattice):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        x = x + self.feature(lattice)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))
    
class group_without_lattice(nn.Module):
    def __init__(self, output_size):
        super(group_without_lattice, self).__init__()
#         self.fc1 = nn.Linear(output_size+9, int(output_size*1.5))  #???? 512+9?
#         self.feature = nn.Linear(9,output_size)
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4),2)
        
    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
#         x = x + self.feature(lattice)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))

class group_with_lattice(nn.Module):
    def __init__(self, output_size):
        super(group_with_lattice, self).__init__()
#         self.fc1 = nn.Linear(output_size+9, int(output_size*1.5))  #???? 512+9?
        self.feature = nn.Linear(9,output_size)
        self.fc1 = nn.Linear(output_size, int(output_size*1.5))
        self.fc2 = nn.Linear(int(output_size*1.5), int(output_size*0.8))
        self.fc3 = nn.Linear(int(output_size*0.8),int(output_size*0.4))
        self.fc4 = nn.Linear(int(output_size*0.4),2)
        
    def forward(self, x,lattice):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        x = x + self.feature(lattice)
        return self.fc4(gelu(self.fc3(gelu(self.fc2(gelu(self.fc1(x)))))))

if __name__ == "__main__":
    a = torch.rand(3,4,5)
    mask = torch.ones(3,4)
    s = Set2Set(5)
    res = s(a,mask)
    print(res.shape)
    # sample = BatchSampler(2,[1,2,3,4,5,6])
    # print(type(sample))
    # print(type(iter(sample)))
    # print(next(iter(sample)))
    # print(next(iter(sample)))
    # print(next(iter(sample)))
    # indices = np.random.permutation(3)
    # batch = []
    # batch.extend(indices)
    # print(len(batch),batch,batch[0].dtype)
    # coords_train_set = np.array([np.array([[1,2,3],[4,4,5],[7,8,9]]),np.array([[1,2,3],[4,4,5],[7,8,9],[10,11,12]]),np.array([[1,2,3],[4,4,5]])])
    # print(coords_train_set.shape)
    # x = torch.rand(3,4,3)
    # batch_index = [0,1,2]
    # test_pos = PositionalEncoding(0.1,10,3)
    # res = test_pos(x,batch_index,0)
    # print(res.shape)
