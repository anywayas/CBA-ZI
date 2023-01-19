import torch
import torch.nn as nn
import torch.nn.functional as F

class Interaction(nn.Module):
    def __init__(self, hidden_size):
        super(Interaction, self).__init__()
        self.hidden_size=hidden_size
        self.dual_att_linear=nn.Linear(3*hidden_size, 1, bias=False)

    def forward(self, encode_input1, encode_input2, input1_mask, input2_mask):
        """
        DynamicCoattentionNetwork
        参考自：https://github.com/PengjieRen/CaSE_RG/blob/master/common/Interaction.py
        (更复杂版本，输入为4维 encode_input1:[B, seq_num, seq_len_q, H])

        input:
        encode_input1: [B, seq_len_q, H]
        encode_input2: [B, seq_len_p, H]
        input1_mask: [B,]
        input2_mask: [B,]
        return:
        [B, seq_len_q, 5*H]
        [B, seq_len_p, 5*H]
        """
        batch_size, seq_len_q, hidden_size = encode_input1.size()
        batch_size, seq_len_p, hidden_size = encode_input2.size()
        E_q = encode_input1
        E_p = encode_input2
        #如果两个输入句子的长度不一样，需要对齐
        E_q_temp=E_q.unsqueeze(1).expand(-1, seq_len_p, -1, -1).contiguous()
        E_p_temp=E_p.unsqueeze(2).expand(-1, -1, seq_len_q, -1).contiguous()
        #三个tensor合并
        E=torch.cat([E_q_temp, E_p_temp, E_q_temp*E_p_temp], dim=-1)#[B, seq_len_p, seq_len_q, 3*H]
        U = self.dual_att_linear(E).squeeze(-1)  # [B, seq_len_p, seq_len_q]

        if input1_mask is not None and input2_mask is not None:
            attentions_mask = torch.einsum("bi,bj->bij", input1_mask, input2_mask)
            attentions_mask = attentions_mask.to(torch.float32)
            U = U.masked_fill(attentions_mask < 0.5, -1e9)

        A_p = F.softmax(U, dim=2)#[B, seq_len_p, seq_len_q]
        B_p = F.softmax(U, dim=1)#[B, seq_len_p, seq_len_q]

        A__p=torch.bmm(A_p, E_q)#[B, seq_len_p, H]
        B__p=torch.bmm(B_p.transpose(1,2), E_p)#[B, seq_len_q, H]

        A___p=torch.bmm(A_p, B__p)#[B, seq_len_p, H]
        B___p=torch.bmm(B_p.transpose(1,2), A__p)#[B, seq_len_q, H]

        G_q_p=torch.cat([E_p, A__p, A___p, E_p*A__p, E_p*A___p], dim=-1)
        G_p_q=torch.cat([E_q, B__p, B___p, E_q*B__p, E_q*B___p], dim=-1)

        return G_p_q, G_q_p


if __name__ == '__main__':
    interaction = Interaction(hidden_size=64)
    encode_input1 = torch.rand(8, 4, 64)
    encode_input2 = torch.rand(8, 4, 64)
    seq1 = torch.tensor([3, 3, 3, 2, 2, 2, 1, 1])
    input1_mask = torch.arange(4).unsqueeze(0).expand(8, -1) < seq1.unsqueeze(1)
    input1_mask = input1_mask.int()
    seq2 = torch.tensor([2, 2, 3, 3, 3, 3, 1, 1])
    input2_mask = torch.arange(4).unsqueeze(0).expand(8, -1) < seq2.unsqueeze(1)
    input2_mask = input2_mask.int()
    G_p_q, G_q_p =interaction.forward(encode_input1, encode_input2, input1_mask, input2_mask)



