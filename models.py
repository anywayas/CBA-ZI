# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_edge_embedding import CBAGNN_Block
from global_variable import TOTAL_ATOM_FEATS


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.score_process = 'mean'
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, interaction_scores, interaction_mask):
        rels = self.rel_emb(rels)  # [B, n_features * n_features]
        rels = torch.nn.functional.normalize(rels, dim=-1)
        heads = torch.nn.functional.normalize(heads, dim=-1)
        tails = torch.nn.functional.normalize(tails, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)  # [B, n_features , n_features]
        # [B,L,H]*[B,H,H]*[B,H,L] = [B,L,L]
        scores = heads @ rels @ tails.transpose(-2, -1)
        if interaction_scores is not None:
            scores = interaction_scores * scores
            if interaction_mask is not None:
                if self.score_process == 'mean':
                    scores = scores * interaction_mask
                else:
                    scores = scores.masked_fill(interaction_mask < 0.5, -1e9)
        if (self.score_process == 'max'):
            scores, _ = scores.reshape(scores.size(0), -1).max(-1)
        if (self.score_process == 'mean'):
            scores = scores.sum(dim=(-2, -1))
            if interaction_mask is not None:
                scores = scores / interaction_mask.sum(dim=(-2, -1))
        if (self.score_process == 'topk'):
            topk_scores = torch.topk(scores.reshape(scores.size(0), -1), 40, -1)
            scores = topk_scores.values.sum(-1)

        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


class Zigzag_updating(nn.Module):
    def __init__(self, hidden_size):
        super(Zigzag_updating, self).__init__()
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.obvious_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, inter_matrix, attention_mask):
        """
        inter_matrix: torch.FloatTensor shaped of [B, k, k, H]
        """
        # B, k1, k2, H
        feat_q = self.wq(inter_matrix.detach())
        feat_k = self.wk(inter_matrix.detach())
        feat_v = self.wv(inter_matrix.detach())
        obvious_matrix = self.obvious_mlp(inter_matrix).squeeze(3)

        def query_single_axis(q, k, v, obv, mask):
            """query single axis
            q,k,v: B, k1, k2, H
            """
            B, k1, k2, H = q.shape
            q = q.reshape(B, -1, H).unsqueeze(2)  # B, k1 * k2, 1, H
            k = k.unsqueeze(2).expand(-1, -1, k2, -1, -1).reshape(B, -1, k2, H)  # B, (k1 * k2), k2, H
            v = v.unsqueeze(2).expand(-1, -1, k2, -1, -1).reshape(B, -1, k2, H)
            obv = obv.unsqueeze(2).expand(-1, -1, k2, -1).reshape(B, -1, 1, k2)  # B, (k1 * k2), 1, k2
            mask = mask.unsqueeze(2).expand(-1, -1, k2, -1).reshape(B, -1, k2)  # B, (k1 * k2), k2
            mask = mask.unsqueeze(2)  # B, (k1 * k2), 1, k2
            qk_logit = torch.einsum("bmqh,bmkh->bmqk", q, k) / (H ** 0.5)  # B, (k1 * k2), 1, k2
            qk_logit = qk_logit * obv
            qk_logit = qk_logit.masked_fill(mask < 0.5, -1e9)
            qk_probs = torch.softmax(qk_logit, dim=3)  # B, (k1 * k2), 1, k2
            ret = torch.einsum("bmqk,bmkh->bmqh", qk_probs, v)
            return ret.reshape(B, k1, k2, H)

        hor = query_single_axis(feat_q, feat_k, feat_v, obvious_matrix, attention_mask)
        ver = query_single_axis(feat_q.transpose(2, 1), feat_k.transpose(2, 1), feat_v.transpose(2, 1), obvious_matrix.transpose(2, 1), attention_mask.transpose(2, 1)).transpose(2, 1)
        return hor + ver


class Zigzag_iteration(nn.Module):
    def __init__(self, hidden_size, r_h_dim):
        super(Zigzag_iteration, self).__init__()
        self.hidden_size = hidden_size
        self.tiny_hidden_size = hidden_size // 8
        self.dual_att_linear_in = nn.Linear(3 * hidden_size, self.tiny_hidden_size, bias=False)
        self.dual_att_linear_out = nn.Linear(3 * hidden_size + self.tiny_hidden_size, 1, bias=False)
        self.zigzag = Zigzag_updating(self.tiny_hidden_size)

        self.layer_norm = nn.LayerNorm(r_h_dim)
        self.left_gate = nn.Linear(r_h_dim, r_h_dim)
        self.right_gate = nn.Linear(r_h_dim, r_h_dim)
        self.out_gate = nn.Linear(r_h_dim, r_h_dim)
        self.to_out = nn.Linear(r_h_dim, r_h_dim)

    def forward(self, encode_input1, encode_input2, attentions_mask):
        """
        :return:
        [B, k, k]
        """
        batch_size, seq_len_q, hidden_size = encode_input1.size()
        batch_size, seq_len_p, hidden_size = encode_input2.size()
        E_q = encode_input1  # B, k, d
        E_p = encode_input2
        E_q_temp = E_q.unsqueeze(1).expand(-1, seq_len_p, -1, -1).contiguous()  # B, k, k, d
        E_p_temp = E_p.unsqueeze(2).expand(-1, -1, seq_len_q, -1).contiguous()  # B, k, k, d

        E = torch.cat([E_q_temp, E_p_temp, E_q_temp * E_p_temp], dim=-1)  # [B, k, k, 3*H]
        E1 = self.dual_att_linear_in(E) # [B, k, k, h]
        M = self.zigzag.forward(E1, attentions_mask)
        U = self.dual_att_linear_out(torch.cat([E, M], -1)).squeeze(-1)  # [B, L, L]

        if attentions_mask is not None:
            U = U.masked_fill(attentions_mask < 0.5, -1e9)
        A_p = F.softmax(U, dim=2)  # [B, k, k]
        B_p = F.softmax(U, dim=1)  # [B, k, k]

        A__p = torch.bmm(A_p, E_q)  # [B, k, H]
        B__p = torch.bmm(B_p.transpose(1, 2), E_p)  # [B, k, H]

        G_q_p = torch.cat([E_p, A__p, E_p * A__p], dim=-1)
        G_p_q = torch.cat([E_q, B__p, E_q * B__p], dim=-1)

        left_gate = torch.sigmoid(self.left_gate(G_q_p))
        right_gate = torch.sigmoid(self.right_gate(G_p_q))
        left = left_gate * G_q_p
        right = right_gate * G_p_q
        out = torch.einsum("bik,bjk->bij", left, right)  # B, k, k
        norm_out = torch.relu(out)
        return norm_out


class RealFakeDDICo(nn.Module):
    def __init__(self, with_global=True, ratio_k=26, dropout_ratio=0):
        super(RealFakeDDICo, self).__init__()
        self.with_global = with_global
        # self.block = SSI_DDI_Block(n_heads=2, in_features=TOTAL_ATOM_FEATS, head_out_feats=64)
        self.block = CBAGNN_Block(in_features=TOTAL_ATOM_FEATS, out_features=128, ratio_k=ratio_k, dropout=dropout_ratio)
        self.gnn_out_hidden_dim = 128
        # self.co_attention = CoAttentionLayer(self.gnn_out_hidden_dim)
        # self.co_attention_softmax = CoAttentionSoftmaxLayer(self.gnn_out_hidden_dim)
        self.rescal = RESCAL(86, self.gnn_out_hidden_dim)
        self.interaction = Zigzag_iteration(hidden_size=128, r_h_dim=128 * 3)

    @staticmethod
    def gen_bias(batch_index):
        l = batch_index.shape[0]
        device = batch_index.device
        shift_batch_index = torch.cat([torch.tensor([-1], dtype=torch.long, device=device), batch_index[:-1]], 0)
        is_start = (batch_index != shift_batch_index).to(torch.long)
        ordinal = torch.arange(l, dtype=torch.long, device=device)
        start = is_start * ordinal
        cum_start = torch.cummax(start, 0)[0]
        bias = ordinal - cum_start
        return bias

    @staticmethod
    def trans(graph_hidden, batch_index):
        batch_size = batch_index.max().cpu().item() + 1
        hidden_dim = graph_hidden.shape[1]
        device = graph_hidden.device

        src = torch.zeros(batch_size, batch_index.shape[0], dtype=torch.long, device=device)
        input_ten = torch.ones(batch_index.shape, dtype=torch.long, device=device)
        src.scatter_add_(0, batch_index.unsqueeze(0), input_ten.unsqueeze(0))
        max_len = src.sum(1).max().cpu().item()

        flat_batch_hidden = torch.zeros(batch_size * max_len, hidden_dim, dtype=torch.float32, device=device)
        flat_efficient_mask = torch.zeros(batch_size * max_len, dtype=torch.long, device=device)
        grade = batch_index * max_len
        bias = RealFakeDDICo.gen_bias(batch_index)
        flat_index = grade + bias

        flat_batch_hidden[flat_index] = flat_batch_hidden[flat_index] + graph_hidden
        flat_efficient_mask[flat_index] = 1

        batch_hidden = flat_batch_hidden.reshape(batch_size, max_len, hidden_dim)
        efficient_mask = flat_efficient_mask.reshape(batch_size, max_len)
        return batch_hidden, efficient_mask

    @staticmethod
    def trans2(graph_hidden, batch_index, global_graph):
        batch_size = batch_index.max().cpu().item() + 1
        hidden_dim = graph_hidden.shape[1]
        device = graph_hidden.device

        src = torch.zeros(batch_size, batch_index.shape[0], dtype=torch.long, device=device)
        input_ten = torch.ones(batch_index.shape, dtype=torch.long, device=device)
        src.scatter_add_(0, batch_index.unsqueeze(0), input_ten.unsqueeze(0))
        max_len = src.sum(1).max().cpu().item()
        # print(max_len)
        #max_len = 40

        flat_batch_hidden = torch.zeros(batch_size * max_len, hidden_dim, dtype=torch.float32, device=device)
        flat_efficient_mask = torch.zeros(batch_size * max_len, dtype=torch.long, device=device)
        grade = batch_index * max_len
        bias = RealFakeDDICo.gen_bias(batch_index)
        flat_index = grade + bias

        flat_batch_hidden[flat_index] = flat_batch_hidden[flat_index] + graph_hidden
        flat_efficient_mask[flat_index] = 1

        batch_hidden = flat_batch_hidden.reshape(batch_size, max_len, hidden_dim)
        efficient_mask = flat_efficient_mask.reshape(batch_size, max_len)

        global_graph = global_graph.unsqueeze(1)  # [B,1,H]
        global_graph_mask = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
        batch_hidden = torch.cat([global_graph, batch_hidden], 1)
        efficient_mask = torch.cat([global_graph_mask, efficient_mask], 1)
        assert batch_hidden.shape[1] == max_len + 1
        return batch_hidden, efficient_mask

    def forward(self, h_data, t_data, rels):
        h_data_atom_feather, h_batch, h_global_graph_emb = self.block(h_data)
        t_data_atom_feather, t_batch, t_global_graph_emb = self.block(t_data)
        if (self.with_global == False):
            # [B, La, H], [B, La]
            drug_a_batch_hidden, drug_a_efficient_mask = RealFakeDDICo.trans(h_data_atom_feather, h_batch)
            # [B, Lb, H], [B, Lb]
            drug_b_batch_hidden, drug_b_efficient_mask = RealFakeDDICo.trans(t_data_atom_feather, t_batch)
        else:
            # [B, La+1, H], [B, La+1]
            drug_a_batch_hidden, drug_a_efficient_mask = RealFakeDDICo.trans2(h_data_atom_feather, h_batch,
                                                                              h_global_graph_emb)
            # [B, Lb+1, H], [B, Lb+1]
            drug_b_batch_hidden, drug_b_efficient_mask = RealFakeDDICo.trans2(t_data_atom_feather, t_batch,
                                                                              t_global_graph_emb)

        attentions_mask = torch.einsum("bi,bj->bij", drug_a_efficient_mask, drug_b_efficient_mask)
        attentions_mask = attentions_mask.to(torch.float32)
        # [B, La, Lb]
        interactions = self.interaction.forward(drug_a_batch_hidden, drug_b_batch_hidden, attentions_mask)

        rel_logit = self.rescal.forward(drug_a_batch_hidden, drug_b_batch_hidden, rels,
                                        interaction_scores=interactions, interaction_mask=attentions_mask)

        rel_probs = torch.sigmoid(rel_logit)
        return rel_logit, rel_probs


