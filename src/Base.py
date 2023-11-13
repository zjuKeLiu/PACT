import torch
from torch import nn
import copy
from Encoder import UnitEncoder,AtomEncoderLayer,MultiHeadAttention,PositionwiseFeedForward,CrystalEncoder,CrystalEncoderLayer
from utils import Set2Set,Embedding, Regression_fun

H = 8 #16
P = 0.3
N = 5
D_MODEL = 64 #512
NUM_ATOM = 96 # 0：PAD_NUM  95: masked_token for pretraining
D_FF = 256 #2048
PAD_NUM = 0


class CrystalTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=D_FF, h=H, dropout=P, num_layer=N, num_atom=NUM_ATOM, pad_num=PAD_NUM):
        super(CrystalTransformer, self).__init__()
        self.Embedding = Embedding(num_atom, d_model)
        self.atten_atom = MultiHeadAttention(h, d_model)
        self.atten_unit = MultiHeadAttention(h, 2*d_model)
        self.ff_atom = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_unit = PositionwiseFeedForward(2*d_model, d_ff, dropout)
        self.unit_encoder = UnitEncoder(AtomEncoderLayer(d_model, copy.deepcopy(self.atten_atom), copy.deepcopy(self.ff_atom), dropout),
                               num_layer, d_model, dropout=0.1)
        self.set2set = Set2Set(d_model, T=3)
#         self.regression_supercon = Regression_fun(2 * d_model)
        self.crystal_encoder = CrystalEncoder(CrystalEncoderLayer(2*d_model, copy.deepcopy(self.atten_unit), copy.deepcopy(self.ff_unit), dropout),
                               num_layer, 2*d_model, dropout=0.1)
        self.regression = Regression_fun(2 * d_model)


    def forward(self, atom_index_data, coords_data,lattice, mask, mask_for_set2set,with_lattice):
        # atom_index_data: batch_size,max_seq_len
        mask = mask.unsqueeze(1)  # batch_size,1,max_seq
        atom_feature = self.Embedding(atom_index_data)  # batch_size,max_seq_len,d_model
        atom_feature = self.unit_encoder(atom_feature, mask, coords_data, lattice)
        unit_feature = self.set2set(atom_feature, mask_for_set2set)
        crystal_feature = self.crystal_encoder(unit_feature, lattice)
        return self.regression(crystal_feature).squeeze()


if __name__ == "__main__":
    fineTuneModel = CrystalTransformer()
    for param in fineTuneModel.parameters():
        if param.dim() > 1:
            nn.init.kaiming_normal_(param)
    fineTunedModel_dic = fineTuneModel.state_dict()
    pretrained_dic = torch.load("model/pretrained.pt")
    common_param_dic = {k:v for k,v in pretrained_dic.items() if k in fineTunedModel_dic}
    fineTunedModel_dic.update(common_param_dic)
    fineTuneModel.load_state_dict(fineTunedModel_dic)
    print(len(common_param_dic))
    print(pretrained_dic["encoder.layers.18.self_atten.linears.0.weight"].device, #cuda:0
          fineTuneModel.state_dict()["encoder.layers.18.self_atten.linears.0.weight"].device) #cpu
    print(fineTuneModel.state_dict()["encoder.layers.18.self_atten.linears.0.weight"] == pretrained_dic["encoder.layers.18.self_atten.linears.0.weight"].cpu())
    print(pretrained_dic["encoder.layers.18.self_atten.linears.0.weight"].device,fineTuneModel.state_dict()["encoder.layers.18.self_atten.linears.0.weight"].device)