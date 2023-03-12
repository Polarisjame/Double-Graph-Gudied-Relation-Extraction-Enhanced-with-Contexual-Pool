from datetime import datetime

import numpy as np
import torch
from dgl import DGLError

def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def logging(s):
    print(datetime.now(), s)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))


def merge_attention(entity_attention_bank, attentions):
    trans_attn = attentions.transpose(0, 1)
    for head in range(trans_attn.size(0)):
        head_attn = trans_attn[head]
        head_attn_avg = torch.mean(head_attn, dim=0)
        entity_attention_bank[head] = head_attn_avg


def entity_cat(edges):
    entity1_h = edges.src['h']
    entity1_attn = edges.src['a']
    entity1_d = edges.src['d']
    entity2_h = edges.dst['h']
    entity2_attn = edges.dst['a']
    entity2_d = edges.dst['d']
    # print(entity1_attn.shape)
    # attn: [edge_num, num_head, s_len]
    A = entity1_attn * entity2_attn
    q = torch.sum(A, dim=1)
    q_t = q / (q.sum(1, keepdim=True) + 1e-5)
    # print(q_t.shape)
    return {'a': q_t,'D':entity1_d,'h_s':entity1_h,'h_o':entity2_h}

def los_sum_exp(mention_feature_list):
    return torch.log(torch.sum(torch.exp(mention_feature_list),dim=0))
    # return torch.mean(mention_feature_list,dim=0)

def merge_mention2entity(g,entity_mention_list,c,d_k):
    n_nodes = g.number_of_nodes()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                try:
                    eid_ij = g.edge_ids(i,j)
                except DGLError:
                    eid_ij = -1
            if eid_ij == -1:
                continue
            c_ij = c[eid_ij].unsqueeze(0)
            mention_head = entity_mention_list[i]
            mention_tail = entity_mention_list[j]

            rate_head =torch.mm(c_ij , mention_head.transpose(0,1)) * d_k
            rate_head = rate_head / (rate_head.sum(1, keepdim=True) + 1e-5)
            rate_tail =torch.mm(c_ij , mention_tail.transpose(0,1)) * d_k
            rate_tail = rate_head / (rate_tail.sum(1, keepdim=True) + 1e-5)

            e_head = torch.mm(rate_head,mention_head)
            e_tail = torch.mm(rate_tail,mention_tail)
            g.edges[eid_ij].data['h_s'] = e_head
            g.edges[eid_ij].data['h_o'] = e_tail
