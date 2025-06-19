# 这是在AKT 模型上加入学生记忆向量的版本

import torch
from torch import nn
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AKT(nn.Module):
    def __init__(self, n_question, student_num,n_pid, d_model, n_blocks, dropout,
                 d_ff=256, kq_same=1, final_fc_dim=512, num_attn_heads=8,
                 separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "akt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        self.student_mem = nn.Embedding(student_num, d_model // num_attn_heads)

        self.model = Architecture(
            n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
            dropout=dropout, d_model=d_model, d_feature=d_model / num_attn_heads,
            d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type,
            student_mem=self.student_mem
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, student_id=None, qtest=False):
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data, student_id)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)
        return (preds, c_reg_loss) if not qtest else (preds, c_reg_loss, concat_q)


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature, d_ff, n_heads,
                 dropout, kq_same, model_type, emb_type, student_mem):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.student_mem = student_mem

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads,
                                 kq_same=kq_same, emb_type=emb_type, student_mem=self.student_mem)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads,
                                 kq_same=kq_same, emb_type=emb_type, student_mem=self.student_mem)
                for _ in range(n_blocks * 2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data, student_id):
        y = qa_embed_data
        x = q_embed_data
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data, student_id=student_id)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False,
                          pdiff=pid_embed_data, student_id=student_id)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True,
                          pdiff=pid_embed_data, student_id=student_id)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, emb_type, student_mem):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same,
            emb_type=emb_type, student_mem=student_mem
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None, student_id=None):
        seqlen = query.size(1)
        nopeek_mask = torch.from_numpy(np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8') == 0).to(device)
        query2 = self.masked_attn_head(query, key, values, mask=nopeek_mask,
                                       zero_pad=(mask == 0), pdiff=pdiff, student_id=student_id)
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, student_mem, bias=True, emb_type="qid"):
        super().__init__()

        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.emb_type = emb_type
        self.student_mem = student_mem
        if emb_type.endswith("avgpool"):
            # pooling
            #self.pool =  nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            pool_size = 3
            self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            # linear
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None, student_id=None):

        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            # v = v.transpose(1,2)
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)#.contiguous().view(bs, -1, self.d_model)
        elif self.emb_type.endswith("linear"):
            # v = v.transpose(1,2)
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)
        elif self.emb_type.startswith("qid"):
            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_model

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff)
            #print("原来的akt", scores.shape)

            if student_id is not None:
                student_bias = self.student_mem(student_id).unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, d_k]
                # print('--------------student_bias-------------------')
                # print(f"student_bias: {student_bias.shape}")
                # print('--------------student_bias-------------------')
                student_bias = self.student_mem(student_id)  # [batch_size, d_k]
                student_bias = student_bias.mean(dim=-1, keepdim=True)  # [batch_size, 1]
                student_bias = student_bias.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
                scores = scores + student_bias  # 自动广播加到 [bs, heads, sl, sl]
                #print("加了学生bias后的akt", scores.shape)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
        return scores

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, emb_type, student_mem):
#         super().__init__()
#         self.d_model = d_model
#         self.d_k = d_feature
#         self.h = n_heads
#         self.kq_same = kq_same
#         self.emb_type = emb_type
#         self.student_mem = student_mem
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, q, k, v, mask, zero_pad, pdiff=None, student_id=None):
#         bs = q.size(0)
#         k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
#         v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
#         q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2) if not self.kq_same else k
#
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#
#         if student_id is not None:
#             student_bias = self.student_mem(student_id).unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, d_k]
#             # print('--------------student_bias-------------------')
#             # print(f"student_bias: {student_bias.shape}")
#             # print('--------------student_bias-------------------')
#             student_bias = self.student_mem(student_id)  # [batch_size, d_k]
#             student_bias = student_bias.mean(dim=-1, keepdim=True)  # [batch_size, 1]
#             student_bias = student_bias.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
#             scores = scores + student_bias  # 自动广播加到 [bs, heads, sl, sl]
#
#
#         scores = scores.masked_fill(mask == 0, -1e32)
#         scores = F.softmax(scores, dim=-1)
#         scores = self.dropout(scores)
#         output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
#         return self.out_proj(output)





def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)  # 结果和上一步一样
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)  # score <0 时，设置为0
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff == None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


# # 这是在AKT 模型上加入学生记忆向量的版本
#
# import torch
# from torch import nn
# import math
# import torch.nn.functional as F
# from enum import IntEnum
# import numpy as np
#
#
# class Dim(IntEnum):
#     batch = 0
#     seq = 1
#     feature = 2
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# class AKT(nn.Module):
#     def __init__(self, n_question, student_num,n_pid, d_model, n_blocks, dropout,
#                  d_ff=256, kq_same=1, final_fc_dim=512, num_attn_heads=8,
#                  separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
#         super().__init__()
#         self.model_name = "akt"
#         self.n_question = n_question
#         self.dropout = dropout
#         self.kq_same = kq_same
#         self.n_pid = n_pid
#         self.l2 = l2
#         self.model_type = self.model_name
#         self.separate_qa = separate_qa
#         self.emb_type = emb_type
#         embed_l = d_model
#
#         if self.n_pid > 0:
#             self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
#             self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
#             self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
#
#         if emb_type.startswith("qid"):
#             self.q_embed = nn.Embedding(self.n_question, embed_l)
#             if self.separate_qa:
#                 self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
#             else:
#                 self.qa_embed = nn.Embedding(2, embed_l)
#
#         self.student_mem = nn.Embedding(student_num, d_model // num_attn_heads)
#
#         self.model = Architecture(
#             n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
#             dropout=dropout, d_model=d_model, d_feature=d_model / num_attn_heads,
#             d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type,
#             student_mem=self.student_mem
#         )
#
#         self.out = nn.Sequential(
#             nn.Linear(d_model + embed_l, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
#             nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
#             nn.Linear(256, 1)
#         )
#         self.reset()
#
#     def reset(self):
#         for p in self.parameters():
#             if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
#                 torch.nn.init.constant_(p, 0.)
#
#     def base_emb(self, q_data, target):
#         q_embed_data = self.q_embed(q_data)
#         if self.separate_qa:
#             qa_data = q_data + self.n_question * target
#             qa_embed_data = self.qa_embed(qa_data)
#         else:
#             qa_embed_data = self.qa_embed(target) + q_embed_data
#         return q_embed_data, qa_embed_data
#
#     def forward(self, q_data, target, pid_data=None, student_id=None, qtest=False):
#         emb_type = self.emb_type
#         if emb_type.startswith("qid"):
#             q_embed_data, qa_embed_data = self.base_emb(q_data, target)
#
#         pid_embed_data = None
#         if self.n_pid > 0:
#             q_embed_diff_data = self.q_embed_diff(q_data)
#             pid_embed_data = self.difficult_param(pid_data)
#             q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
#             qa_embed_diff_data = self.qa_embed_diff(target)
#             if self.separate_qa:
#                 qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
#             else:
#                 qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
#             c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
#         else:
#             c_reg_loss = 0.
#
#         d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data, student_id)
#         concat_q = torch.cat([d_output, q_embed_data], dim=-1)
#         output = self.out(concat_q).squeeze(-1)
#         preds = torch.sigmoid(output)
#         return (preds, c_reg_loss) if not qtest else (preds, c_reg_loss, concat_q)
#
#
# class Architecture(nn.Module):
#     def __init__(self, n_question, n_blocks, d_model, d_feature, d_ff, n_heads,
#                  dropout, kq_same, model_type, emb_type, student_mem):
#         super().__init__()
#         self.d_model = d_model
#         self.model_type = model_type
#         self.student_mem = student_mem
#
#         if model_type in {'akt'}:
#             self.blocks_1 = nn.ModuleList([
#                 TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
#                                  d_ff=d_ff, dropout=dropout, n_heads=n_heads,
#                                  kq_same=kq_same, emb_type=emb_type, student_mem=self.student_mem)
#                 for _ in range(n_blocks)
#             ])
#             self.blocks_2 = nn.ModuleList([
#                 TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
#                                  d_ff=d_ff, dropout=dropout, n_heads=n_heads,
#                                  kq_same=kq_same, emb_type=emb_type, student_mem=self.student_mem)
#                 for _ in range(n_blocks * 2)
#             ])
#
#     def forward(self, q_embed_data, qa_embed_data, pid_embed_data, student_id):
#         y = qa_embed_data
#         x = q_embed_data
#         for block in self.blocks_1:
#             y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data, student_id=student_id)
#         flag_first = True
#         for block in self.blocks_2:
#             if flag_first:
#                 x = block(mask=1, query=x, key=x, values=x, apply_pos=False,
#                           pdiff=pid_embed_data, student_id=student_id)
#                 flag_first = False
#             else:
#                 x = block(mask=0, query=x, key=x, values=y, apply_pos=True,
#                           pdiff=pid_embed_data, student_id=student_id)
#                 flag_first = True
#         return x
#
#
# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, emb_type, student_mem):
#         super().__init__()
#         self.masked_attn_head = MultiHeadAttention(
#             d_model, d_feature, n_heads, dropout, kq_same=kq_same,
#             emb_type=emb_type, student_mem=student_mem
#         )
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ff, d_model)
#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, mask, query, key, values, apply_pos=True, pdiff=None, student_id=None):
#         seqlen = query.size(1)
#         nopeek_mask = torch.from_numpy(np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8') == 0).to(device)
#         query2 = self.masked_attn_head(query, key, values, mask=nopeek_mask,
#                                        zero_pad=(mask == 0), pdiff=pdiff, student_id=student_id)
#         query = query + self.dropout1(query2)
#         query = self.layer_norm1(query)
#         if apply_pos:
#             query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
#             query = query + self.dropout2(query2)
#             query = self.layer_norm2(query)
#         return query
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, emb_type, student_mem):
#         super().__init__()
#         self.d_model = d_model
#         self.d_k = d_feature
#         self.h = n_heads
#         self.kq_same = kq_same
#         self.emb_type = emb_type
#         self.student_mem = student_mem
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, q, k, v, mask, zero_pad, pdiff=None, student_id=None):
#         bs = q.size(0)
#         k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
#         v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
#         q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2) if not self.kq_same else k
#
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#
#         if student_id is not None:
#             student_bias = self.student_mem(student_id).unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, d_k]
#             # print('--------------student_bias-------------------')
#             # print(f"student_bias: {student_bias.shape}")
#             # print('--------------student_bias-------------------')
#             student_bias = self.student_mem(student_id)  # [batch_size, d_k]
#             student_bias = student_bias.mean(dim=-1, keepdim=True)  # [batch_size, 1]
#             student_bias = student_bias.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
#             scores = scores + student_bias  # 自动广播加到 [bs, heads, sl, sl]
#
#
#         scores = scores.masked_fill(mask == 0, -1e32)
#         scores = F.softmax(scores, dim=-1)
#         scores = self.dropout(scores)
#         output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
#         return self.out_proj(output)
#
#
#
# def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
#     """
#     This is called by Multi-head atention object to find the values.
#     """
#     # d_k: 每一个头的dim
#     scores = torch.matmul(q, k.transpose(-2, -1)) / \
#              math.sqrt(d_k)  # BS, 8, seqlen, seqlen
#     bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
#
#     x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
#     x2 = x1.transpose(0, 1).contiguous()
#
#     with torch.no_grad():
#         scores_ = scores.masked_fill(mask == 0, -1e32)
#         scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
#         scores_ = scores_ * mask.float().to(device)  # 结果和上一步一样
#         distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
#         disttotal_scores = torch.sum(
#             scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
#         # print(f"distotal_scores: {disttotal_scores}")
#         position_effect = torch.abs(
#             x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
#         # bs, 8, sl, sl positive distance
#         dist_scores = torch.clamp(
#             (disttotal_scores - distcum_scores) * position_effect, min=0.)  # score <0 时，设置为0
#         dist_scores = dist_scores.sqrt().detach()
#     m = nn.Softplus()
#     gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
#     # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
#     if pdiff == None:
#         total_effect = torch.clamp(torch.clamp(
#             (dist_scores * gamma).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
#     else:
#         diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
#         diff = diff.sigmoid().exp()
#         total_effect = torch.clamp(torch.clamp(
#             (dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
#     scores = scores * total_effect
#
#     scores.masked_fill_(mask == 0, -1e32)
#     scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
#     # print(f"before zero pad scores: {scores.shape}")
#     # print(zero_pad)
#     if zero_pad:
#         pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
#         scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
#     # print(f"after zero pad scores: {scores}")
#     scores = dropout(scores)
#     output = torch.matmul(scores, v)
#     # import sys
#     # sys.exit()
#     return output
#
#
# class LearnablePositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = 0.1 * torch.randn(max_len, d_model)
#         pe = pe.unsqueeze(0)
#         self.weight = nn.Parameter(pe, requires_grad=True)
#
#     def forward(self, x):
#         return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
#
#
# class CosinePositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = 0.1 * torch.randn(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.weight = nn.Parameter(pe, requires_grad=False)
#
#     def forward(self, x):
#         return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)