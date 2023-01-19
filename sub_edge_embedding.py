import torch
import numpy as np
from torch import nn
import random
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.data import Data,Batch
from torch_geometric.nn import (GATConv,
                                GINConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool)
from global_variable import MOL_EDGE_LIST_FEAT_MTX_PLUS, ALL_TRUE_H_WITH_TR, ALL_TRUE_T_WITH_HR
from global_variable import ALL_TAIL_PER_HEAD, ALL_HEAD_PER_TAIL, TOTAL_ATOM_FEATS

class DrugDataset(Dataset):
    def __init__(self, tri_list, all_drug_old_ids, all_drug_new_ids, neg_ent=1, mode="none"):
        self.tri_list = tri_list
        self.neg_ent = neg_ent
        self.mode = mode
        # old drug
        self.old_drug_ids = all_drug_old_ids
        self.new_drug_ids = all_drug_new_ids
    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn_fake_or_real(self, batch):
        rels = []
        h_samples = []
        t_samples = []
        label = []

        for h, t, r, l in batch:
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            h_samples.append(h_data)
            t_samples.append(t_data)
            rels.append(r)
            label.append(l)

        if self.mode == "ttrain":
            # 随机生成负样本,标签为 0
            for h, t, r, l in batch:
                neg_heads, neg_tails = self.normal_batch(h, t, r, self.neg_ent)

                for neg_h in neg_heads:
                    h_samples.append(self.__create_graph_data(neg_h))
                    t_samples.append(t_data)
                    rels.append(r)
                    label.append(0)

                for neg_t in neg_tails:
                    h_samples.append(h_data)
                    t_samples.append(self.__create_graph_data(neg_t))
                    rels.append(r)
                    label.append(0)

        h_samples = Batch.from_data_list(h_samples)
        t_samples = Batch.from_data_list(t_samples)
        rels = torch.LongTensor(rels)
        label = torch.LongTensor(label)

        tri = (h_samples, t_samples, rels, label)

        return tri

    def __create_graph_data(self, id):
        features = MOL_EDGE_LIST_FEAT_MTX_PLUS[id][0]
        edge_index = MOL_EDGE_LIST_FEAT_MTX_PLUS[id][1]
        edge_feats = MOL_EDGE_LIST_FEAT_MTX_PLUS[id][2]
        edge_type = MOL_EDGE_LIST_FEAT_MTX_PLUS[id][3]

        return Data(x=features, edge_index=edge_index, edge_attr= edge_feats, edge_type = edge_type)

    #以下四个函数用于随机生成负样本
    def __corrupt_ent(self, other_ent, r, other_ent_with_r_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            if(self.mode == "ttrain"):
                candidates = np.random.choice(self.old_drug_ids, (max_num - current_size) * 2)
            elif (self.mode == "ttest"):
                candidates = np.random.choice(self.new_drug_ids, (max_num - current_size) * 2)
            mask = np.isin(candidates, other_ent_with_r_dict[(other_ent, r)], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])

        if corrupted_ents:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])

    def __corrupt_head(self, t, r, n=1):
        return self.__corrupt_ent(t, r, ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h, r, n=1):
        return self.__corrupt_ent(h, r, ALL_TRUE_T_WITH_HR, n)

    def normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        prob = ALL_TAIL_PER_HEAD[r] / (ALL_TAIL_PER_HEAD[r] + ALL_HEAD_PER_TAIL[r])
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1
        return (self.__corrupt_head(t, r, neg_size_h),
                self.__corrupt_tail(h, r, neg_size_t))

    def generate_neg_pair(self, pos_pairs):
        neg_pairs = []
        for h, t, r, l in pos_pairs:
            neg_heads, neg_tails = self.normal_batch(h, t, r, 1)

            for neg_h in neg_heads:
                tup = (neg_h, t, r, 0)
                neg_pairs.append(tup)

            for neg_t in neg_tails:
                tup = (h, neg_t, r, 0)
                neg_pairs.append(tup)
        return neg_pairs
class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

class CBAGNN_Block(nn.Module):
    def __init__(self, in_features, out_features, ratio_k, dropout):
        super().__init__()
        self.n_heads = 2
        self.in_features = in_features
        self.out_features = out_features
        self.temp_features = int(self.out_features/self.n_heads)
        self.conv1 = RGCNConv(in_channels=TOTAL_ATOM_FEATS, out_channels =out_features, num_relations=16)
        self.conv2 = GATConv(self.out_features, self.temp_features, self.n_heads)
        self.conv3 = GATConv(self.out_features, self.temp_features, self.n_heads)
        self.conv4 = GATConv(self.out_features, self.temp_features, self.n_heads)

        self.readout = SAGPooling(in_channels=out_features, ratio=ratio_k)
        self.initial_norm = LayerNorm(self.in_features)
        self.second_norm = LayerNorm(out_features)
        self.third_norm = LayerNorm(out_features)
        self.forth_norm = LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)  #原来是0.2
        #self.pool = TopKPooling(128, ratio=0.5)

    def forward(self, data):
        data.x = self.initial_norm(data.x, data.batch)
        data.x = self.conv1(data.x, data.edge_index, data.edge_type)
        data.x = F.elu(self.second_norm(data.x, data.batch))
        data.x = self.dropout(data.x)
        data.x = self.conv2(data.x, data.edge_index)
        data.x = F.elu(self.third_norm(data.x, data.batch))
        data.x = self.dropout(data.x)
        data.x = self.conv3(data.x, data.edge_index)
        #data.x = F.elu(self.forth_norm(data.x, data.batch))
        # data.x = self.conv4(data.x, data.edge_index)

        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(data.x, data.edge_index, batch=data.batch)
        global_graph_emb = global_add_pool(att_x, att_batch) # global_graph_emb:(batchsize,64)
        return att_x, att_batch, global_graph_emb #att_x: [K,hidden]



