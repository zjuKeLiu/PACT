import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import time
from torch.autograd import Variable
import copy
import random
import os
import csv
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from Base import CrystalTransformer
from utils import BatchSampler,pad_to_same
# from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
SEED = 126
PAD_NUM = 0
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

root_path = "/home/liuke/liuke/prj/score/CrystalTransformer/dataset/formation_energy"
dataset_path = f"{root_path}/labels.csv"
coord_path = f"{root_path}/real_coord.npy"
data_path = f"{root_path}/molecule.npy"
lattice_path = f"{root_path}/lattice_new.npy"

mol = np.load(file=data_path, allow_pickle=True)
coords = np.load(file=coord_path, allow_pickle=True)
lattice = np.load(file=lattice_path, allow_pickle=True)
data = pd.read_csv(dataset_path).to_numpy()
# labels = pd.read_csv(dataset_path)
# labels = labels["supercon"][:].values
labels, labelsF = data[:,0],data[:,1].astype(float)
dist = np.arange(0, 5.12, 0.01)
# for i in range(coords.shape[0]):
#     new_dis = np.zeros((coords[i].shape[0], 512))
#     for j, dis in enumerate(dist):
#         temp = coords[i] - coords[i].mean(0)
#         new_dis[:,j] = np.exp(-((np.sqrt(np.sum(temp**2, axis=1))-dis)**2)/0.25)
#     coords[i] = new_dis

# original_mol_train_set, mol_test_set, original_coords_train_set, coords_test_set, original_labels_train_set, labels_test_set, original_lattice_train_set, lattice_test_set = train_test_split(mol, coords, labels, lattice, test_size=0.15, random_state=42)
# mol_train_set, mol_test_set,coords_train_set, coords_test_set, labels_train_set, labels_test_set, lattice_train_set, lattice_test_set = train_test_split(mol, coords, labels, lattice, test_size=0.2, random_state=42)
mol_train_set, mol_test_set,coords_train_set, coords_test_set, labels_train_set, labels_test_set,labelsF_train_set,labelsF_test_set, lattice_train_set, lattice_test_set = train_test_split(mol, coords, labels,labelsF,lattice, test_size=0.2, random_state=42)
# mol_test_set, mol_eval_set, coords_test_set, coords_eval_set, labels_test_set, labels_eval_set, lattice_test_set, lattice_eval_set = train_test_split(mol_test_set, coords_test_set, labels_test_set, lattice_test_set, test_size=0.5, random_state=34)
val = list()
tra = list()

def binary_acc(preds, y):
    count = 0
    _, p = preds.max(1)
    for i,j in zip(p, y):
        if i == j:
            count += 1
    return count/len(y)

EVALUATE = nn.CrossEntropyLoss()
criteon = nn.CrossEntropyLoss()
with_lattice = False
MSE = nn.MSELoss()
MAE = nn.L1Loss()

def train(model, iterator, optimizer, criteon):
    avg_lossF = []
    model.train()  # 表示进入训练模式

    for i, batch_index in tqdm(enumerate(iterator)):
        #print(i)
        #train_data = pad_to_same(mol_train_set[batch_index], PAD_NUM).long().cuda()
        atom_index_data = pad_to_same(mol_train_set[batch_index], PAD_NUM).long().cuda()
        #########################cuda()$$$$$$$$$$$$$$$$$$$###########
        #atom_index_data = pad_to_same(mol_train_set[batch_index], PAD_NUM).long()
        coords_data = coords_train_set[batch_index]
        #lattice = torch.tensor(lattice_train_set[batch_index],dtype=torch.float32)
        lattice = lattice_train_set[batch_index]
        #print(atom_index_data.shape)
        #print(coords_data.size,atom_index_data.size())

        mask_for_set2set = (atom_index_data == PAD_NUM).type(torch.bool)
        mask = 1 - (atom_index_data == PAD_NUM).float()
        #print(train_data.shape,mask.shape)
        fromation_energy = model(atom_index_data, coords_data, lattice, mask, mask_for_set2set, with_lattice)
        #print(fromation_energy.shape)
        #########################cuda()$$$$$$$$$$$$$$$$$$$###########
        loss2 = MAE(fromation_energy,torch.tensor(labelsF_train_set[batch_index],dtype = torch.float32).cuda())
        #loss2 = MAE(fromation_energy,torch.tensor(labelsF_train_set[batch_index],dtype = torch.float32))
        
        #print(loss2)
        avg_lossF.append(loss2.item())

        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
        del atom_index_data

    avg_lossF = np.array(avg_lossF).mean()
    return avg_lossF


# 评估函数
def test(model, iterator, criteon):
    avg_lossF = []
    model.eval()  # 表示进入测试模式

    with torch.no_grad():
        for batch_index in iterator:
            ################################cuda()#########################################
            atom_index_data = pad_to_same(mol_test_set[batch_index], PAD_NUM).long().cuda()
            #atom_index_data = pad_to_same(mol_test_set[batch_index], PAD_NUM).long()
            coords_data = coords_test_set[batch_index]
            mask_for_set2set = (atom_index_data == PAD_NUM).type(torch.bool)
            mask = 1 - (atom_index_data == PAD_NUM).float()
            # print(train_data.shape,mask.shape)
            #lattice = torch.tensor(lattice_test_set[batch_index],dtype=torch.float32)
            lattice = lattice_test_set[batch_index]
            fromation_energy = model(atom_index_data, coords_data,lattice ,mask, mask_for_set2set,with_lattice)
            loss2 = MAE(fromation_energy,torch.tensor(labelsF_test_set[batch_index],dtype = torch.float32).cuda())
            ####################################cuda()###########################################
            
            #loss2 = MAE(fromation_energy,torch.tensor(labelsF_test_set[batch_index],dtype = torch.float32))
            avg_lossF.append(loss2.item())

    avg_lossF = np.array(avg_lossF).mean()
    return avg_lossF


# define the model and assign some hyper-parameters
best_MAE = 100000000
CryT = CrystalTransformer(num_layer=5)
optimizer = optim.Adam(CryT.parameters(), lr=LEARNING_RATE)#, weight_decay=LEARNING_RATE/100)
val = list()
tra = list()
#####################################cuda()#######################
CryT.float().cuda()
#CryT.float()


for epoch in range(200):

    train_batch = BatchSampler(BATCH_SIZE, mol_train_set)
    test_batch = BatchSampler(BATCH_SIZE, mol_test_set)

    train_iterator = iter(train_batch)
    start_time = time.time()
    loss_F = train(CryT, train_iterator, optimizer, criteon)
    end_time = time.time()
    
    test_iterator = iter(test_batch)
    test_loss_F = test(CryT, test_iterator, criteon)
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)


    if test_loss_F < best_MAE:  # 只要模型效果变好，就保存
        best_MAE = test_loss_F
        torch.save(CryT, '/home/liuke/liuke/prj/score/CrystalTransformer/multi_hierachy/test/fe/best.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
    print(f'\t Train_lossF: {loss_F:.3f}')
    print(f'\t Test_lossF: {test_loss_F:.3f}')
    print(f'\t Best MAE: {best_MAE}')

