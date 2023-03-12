import copy
from math import ceil
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
from transformers import *

from utils import get_cuda, merge_attention, entity_cat, los_sum_exp
from models.TransformerEncoder import Encoder


class GAIN_GloVe(nn.Module):
    def __init__(self, config):
        super(GAIN_GloVe, self).__init__()
        self.config = config

        word_emb_size = config.word_emb_size
        vocabulary_size = config.vocabulary_size
        encoder_input_size = word_emb_size
        self.activation = nn.Tanh() if config.activation == 'tanh' else nn.ReLU()

        self.word_emb = nn.Embedding(vocabulary_size, word_emb_size, padding_idx=config.word_pad)
        if config.pre_train_word:
            self.word_emb = nn.Embedding(config.data_word_vec.shape[0], word_emb_size, padding_idx=config.word_pad)
            self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec[:, :word_emb_size]))

        self.word_emb.weight.requires_grad = config.finetune_word
        if config.use_entity_type:
            encoder_input_size += config.entity_type_size
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)

        if config.use_entity_id:
            encoder_input_size += config.entity_id_size
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        self.encoder = BiLSTM(encoder_input_size, config)

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == 2 * config.lstm_hidden_size, 'gcn dim should be the lstm hidden dim * 2'
        rel_name_lists = ['intra', 'inter', 'global']
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        self.bank_size = self.config.gcn_dim * (self.config.gcn_layers + 1)
        self.mention_encoder = Encoder(self.bank_size, config.encode_layer)
        self.dropout = nn.Dropout(self.config.dropout)

        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 5 + self.gcn_dim * 4, self.bank_size * 2),  #
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_dim, edge_feat=self.gcn_dim,
                                       activation=self.activation, dropout=config.dropout)

        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)
        self.attention = Attention(self.bank_size * 2, self.gcn_dim * 4)

    def forward(self, **params):
        '''
            words: [batch_size, max_length]
            src_lengths: [batchs_size]
            mask: [batch_size, max_length]
            entity_type: [batch_size, max_length]
            entity_id: [batch_size, max_length]
            mention_id: [batch_size, max_length]
            distance: [batch_size, max_length]
            entity2mention_table: list of [local_entity_num, local_mention_num]
            graphs: list of DGLHeteroGraph
            h_t_pairs: [batch_size, h_t_limit, 2]
        '''
        src = self.word_emb(params['words'])
        mask = params['mask']
        bsz, slen, _ = src.size()

        if self.config.use_entity_type:
            src = torch.cat([src, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            src = torch.cat([src, self.entity_id_emb(params['entity_id'])], dim=-1)

        # src: [batch_size, slen, encoder_input_size]
        # src_lengths: [batchs_size]

        encoder_outputs, (output_h_t, _) = self.encoder(src, params['src_lengths'])
        encoder_outputs[mask == 0] = 0
        # encoder_outputs: [batch_size, slen, 2*encoder_hid_size]
        # output_h_t: [batch_size, 2*encoder_hid_size]

        graphs = params['graphs']

        mention_id = params['mention_id']
        features = None

        for i in range(len(graphs)):
            encoder_output = encoder_outputs[i]  # [slen, 2*encoder_hid_size]
            mention_num = torch.max(mention_id[i])
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen]
            # average word -> mention
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            x = torch.mm(select_metrix, encoder_output)  # [mention_num, 2*encoder_hid_size]

            x = torch.cat((output_h_t[i].unsqueeze(0), x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        graph_big = dgl.batch_hetero(graphs)
        output_features = [features]

        for GCN_layer in self.GCN_layers:
            features = GCN_layer(graph_big, {"node": features})["node"]  # [total_mention_nums, gcn_dim]
            output_features.append(features)

        output_feature = torch.cat(output_features, dim=-1)

        graphs = dgl.unbatch_hetero(graph_big)

        # mention -> entity
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id'])
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        global_info = get_cuda(torch.Tensor(bsz, self.bank_size))

        cur_idx = 0
        entity_graph_feature = None
        # for i in range(len(graphs)):
        #     # average mention -> entity
        #     select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]
        #     select_metrix[0][0] = 1
        #     mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
        #     select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
        #     node_num = graphs[i].number_of_nodes('node')
        #     entity_representation = torch.mm(select_metrix, output_feature[cur_idx:cur_idx + node_num])
        #     entity_bank[i, :select_metrix.size(0) - 1] = entity_representation[1:]
        #     global_info[i] = output_feature[cur_idx]
        #     cur_idx += node_num
        #
        #     if entity_graph_feature is None:
        #         entity_graph_feature = entity_representation[1:, -self.config.gcn_dim:]
        #     else:
        #         entity_graph_feature = torch.cat(
        #             (entity_graph_feature, entity_representation[1:, -self.config.gcn_dim:]), dim=0)
        for i in range(len(graphs)):
            # average mention -> entity
            select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]
            select_metrix = select_metrix[1:].unsqueeze(-1).expand([-1, -1, self.bank_size])
            node_num = graphs[i].number_of_nodes('node')
            e2ms = torch.mul(select_metrix, output_feature[cur_idx + 1:cur_idx + node_num])
            nonZeroRows = torch.abs(e2ms).sum(dim=-1) > 0
            entity_representation = self.mention_encoder(e2ms, nonZeroRows)
            entity_bank[i, :select_metrix.size(0) - 1] = entity_representation
            global_info[i] = output_feature[cur_idx]
            cur_idx += node_num

            if entity_graph_feature is None:
                entity_graph_feature = entity_representation[:, -self.config.gcn_dim:]
            else:
                entity_graph_feature = torch.cat(
                    (entity_graph_feature, entity_representation[:, -self.config.gcn_dim:]), dim=0)
        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)

        # [batch_size, h_t_limit, bank_size]
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1)

        entity_graphs = params['entity_graphs']
        entity_graph_big = dgl.batch(entity_graphs)
        self.edge_layer(entity_graph_big, entity_graph_feature)
        entity_graphs = dgl.unbatch(entity_graph_big)
        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table']
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(h_t_limit):
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break

                h = h_t_pairs[i, j, 0].item()
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue

                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].all_edges())
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0:
                    continue

                # forward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v))
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')

        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )

        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))
        return predictions


class GAIN_BERT(nn.Module):
    def __init__(self, config):
        super(GAIN_BERT, self).__init__()
        self.config = config
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        if config.use_entity_type:
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)
        if config.use_entity_id:
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.hid_size = config.bert_hid_size
        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size
        self.gcn_out_dim = config.bert_hid_size + config.entity_id_size + config.entity_type_size
        # self.trans_before_gcn = nn.Linear(self.gcn_dim,self.gcn_out_dim)
        rel_name_lists = ['intra', 'inter', 'global']
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        self.bank_size = self.gcn_out_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 5 + self.gcn_out_dim * 4, self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_out_dim, edge_feat=self.gcn_out_dim,
                                       activation=self.activation, dropout=config.dropout)

        self.path_info_mapping = nn.Linear(self.gcn_out_dim * 4, self.gcn_out_dim * 4)

        self.attention = Attention(self.bank_size * 2, self.gcn_out_dim * 4)
        # self.TransformerEncoder = TransformerEncoder(d_model=self.bank_size, num_layer=8, nhead=4,
        #                                              output_size=self.bank_size)
        self.softmax = nn.Softmax(-1)
    def forward(self, **params):
        """
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        ht_pair_distance: [batch_size, h_t_limit]
        """
        words = params['words']
        mask = params['mask']
        bsz, slen = words.size()
        output = self.bert(input_ids=words, attention_mask=mask, output_attentions=True)
        encoder_outputs = output.last_hidden_state
        sentence_cls = output.pooler_output
        attention = output.attentions
        # encoder_outputs, sentence_cls = self.bert(input_ids=words, attention_mask=mask)
        # encoder_outputs[mask == 0] = 0
        # encoder_outputs: (batch_size, sequence_length, hidden_size) 每个单词的Embedding
        # sentence_cls: (batch_size, hidden_size) 文档Embedding
        # attention: (batch_size, num_Layers, num_head, sen_len, sen_len) Transformer所有层的Attention

        # ---------------------------------------------------------------------------------------------------------
        # 以上利用BERT分别得到了每个单词的编码 encoder_outputs 以及 文档整体编码 sentence_cls和Attention矩阵last_attention
        # ---------------------------------------------------------------------------------------------------------
        encoder_outputs_origin = get_cuda(torch.Tensor(bsz,slen,self.hid_size))
        encoder_outputs_origin[:,:,:] = encoder_outputs[:,:,:]
        if self.config.use_entity_type:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)

        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)

        graphs = params['graphs']
        mention_id = params['mention_id']
        token_id = params['token_id']
        features = None
        attentions = None

        last_attention = attention[-1]  # [batch_size, num_heads, sen_len, sen_len]
        # d = encoder_outputs.reshape(encoder_outputs.size()[0] * encoder_outputs.size()[1], -1)  # [batch_size*slen, bert_hid]
        # ---------------------------------------------------------------------------------------------------------
        # 以下将原来平均池化得到mention特征改为提取special_token的特征
        # ---------------------------------------------------------------------------------------------------------
        # cur_device = torch.cuda.current_device()
        # graph_range1 = ceil(len(graphs)/3)*(cur_device-1)
        # graph_range2 = min(ceil(len(graphs)/3)*(cur_device),len(graphs))
        # print(words.shape)
        # print(len(graphs),len(params['entity2mention_table']),len(params['entity_graphs']))
        # print(graph_range1,graph_range2)
        # graphs = graphs[graph_range1:graph_range2]
        for i in range(len(graphs)):  # 分batch遍历
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]
            mention_num = torch.max(mention_id[i])
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            
            mentions = token_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen] 第i行表示第i个提及在该文档出现的位置处标1
            
            # mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            # select_metrix = (mention_index == mentions).float()  # [mention_num, slen] 第i行表示第i个提及在该文档出现的位置处标1
            # word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1,
            #                                                                            slen)  # [mention_num,slen] 每行标1个数
            # select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers,
            #                             select_metrix)  # 平均化

            x = torch.mm(select_metrix, encoder_output)  # [mention_num, bert_hid] 得到每个提及的embedding
            # # mention Attention
            # mention_contexcual_vecs = get_cuda(torch.Tensor(mention_num, self.hid_size))
            # d = encoder_outputs_origin[i]
            # for j in range(mention_num):
            #     mention = select_metrix[j]
            #     mention_pos = torch.nonzero(mention, as_tuple=False)
            #     if len(mention_pos) == 0:
            #         mention_contexcual_vecs[j] = get_cuda(torch.zeros(self.hid_size))
            #         continue
            #     # pos = torch.min(mention_pos)
            #     pos1 = torch.min(mention_pos)
            #     pos2 = torch.max(mention_pos)
            #     mention_embedding = d[pos1:pos2+1]

            #     # a = torch.mm(mention_embedding.unsqueeze(0), d.T)
            #     a = torch.mm(mention_embedding, d.T)
            #     a = self.softmax(a)
            #     a = torch.mean(a, dim=0,keepdim=True)
            #     c = torch.mm(a, d)
            #     mention_contexcual_vecs[j] = c
            # a = torch.mm(sentence_cls[i][0:d.shape[-1]].unsqueeze(0), d.T)
            # a = self.softmax(a)
            # c = torch.mm(a, d)
            # c = torch.cat((sentence_cls[i],c.squeeze(0)),dim=-1)
            # x = torch.cat((x, mention_contexcual_vecs), dim=-1)
            # x = torch.cat((c.unsqueeze(0), x), dim=0)  # 文档节点作为0号节点
            attn = torch.matmul(select_metrix, last_attention[i])  # [num_heads, mention_num, sen_len]
            attn = attn.transpose(0, 1)  # [mention_num, num_heads, sen_len]
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0)  # 文档节点作为0号节点

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)  # 收集每个batch的提及节点embedding，按行插入
            
            if attentions is None:
                attentions = attn
            else:
                attentions = torch.cat((attentions, attn), dim=0)

        graph_big = dgl.batch_hetero(graphs)
        output_features = [features]  # 收集每层GCN的特征，排除文档节点

        # ---------------------------------------------------------------------------------------------------------
        # 下面的循环是mention图的多层GCN，具体GCN的实现在RelGraphConvLayer中
        # ---------------------------------------------------------------------------------------------------------

        for GCN_layer in self.GCN_layers:
            features = GCN_layer(graph_big, {"node": features})["node"]  # [total_mention_nums, gcn_dim]
            output_features.append(features)

        output_feature = torch.cat(output_features, dim=-1)  # 拼接每层特征
        graphs = dgl.unbatch_hetero(graph_big)

        # mention -> entity

        # ---------------------------------------------------------------------------------------------------------
        # 下面实现了mention到entity特征的转换并赋值给了实体图中的节点,同时将mention的attention矩阵融合到entity
        # ---------------------------------------------------------------------------------------------------------

        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        # entity2mention_table = entity2mention_table[graph_range1:graph_range2]
        entity_num = torch.max(params['entity_id'])
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        entity_attention_bank = get_cuda(torch.Tensor(entity_num, last_attention.size(1), slen))  # [entity_num, num_head, s_len]
        global_info = get_cuda(torch.Tensor(bsz, self.bank_size))

        cur_idx = 0
        entity_graph_feature = None
        entity_graph_attention = None
        context_output = None
        entity_idx = 0
        entity_mention_list = {}

        for i in range(len(graphs)):
            # average mention -> entity
            local_entity_num = len(entity2mention_table[i])  # 这里local_entity_num包含了一个文档节点
            node_num = graphs[i].number_of_nodes('node')
            encoder_output = encoder_outputs[i]
            global_info[i] = output_feature[cur_idx]
            # batch中的第i个文档，第j个entity
            for j in range(1, local_entity_num):
                mention_num = int(torch.sum(entity2mention_table[i][j]))
                mention_feature_list = get_cuda(torch.Tensor(mention_num, self.bank_size))
                mention_attention_list = get_cuda(torch.Tensor(mention_num, last_attention.size(1), slen))
                row = 0
                # print(mention_num)
                for k in range(1, node_num):
                    if entity2mention_table[i][j][k] == 1:
                        mention_feature_list[row][:] = output_feature[cur_idx + k]
                        mention_attention_list[row][:][:] = attentions[cur_idx + k - 1 - i]# 得到该entity所有mention的attention矩阵 [mention_num, num_head, s_len]
                        row += 1  
                if row > 0:
                    # entity_bank[i][j - 1] = self.TransformerEncoder(mention_feature_list)
                    entity_bank[i][j - 1] = los_sum_exp(mention_feature_list)  # 保存entity节点的特征
                    entity_mention_list[entity_idx] = mention_feature_list[:,-self.gcn_out_dim:]
                    merge_attention(entity_attention_bank[j - 1], mention_attention_list)
                else:
                    entity_bank[i][j - 1] = get_cuda(torch.zeros(self.bank_size)) 
                    entity_mention_list[entity_idx] = torch.zeros(self.bank_size)
                    entity_attention_bank[j - 1] = get_cuda(torch.zeros(last_attention.size(1), slen)) 
                entity_idx += 1

            cur_idx += node_num
            if entity_graph_feature is None:
                entity_graph_feature = entity_bank[i][:local_entity_num-1,-self.gcn_out_dim:]
            else:
                entity_graph_feature = torch.cat((entity_graph_feature, entity_bank[i][:local_entity_num-1,-self.gcn_out_dim:]),
                                                 dim=0)
            if entity_graph_attention is None:
                entity_graph_attention = entity_attention_bank[:local_entity_num-1]
            else:
                entity_graph_attention = torch.cat((entity_graph_attention, entity_attention_bank[:local_entity_num-1]),
                                                   dim=0)  # [num_nodes, num_head, s_len]

            if context_output is None:
                context_output = encoder_output.unsqueeze(0).expand(local_entity_num - 1, -1, -1)
            else:
                context_output = torch.cat(
                    (context_output, encoder_output.unsqueeze(0).expand(local_entity_num - 1, -1, -1)),
                    dim=0)
        
        # ---------------------------------------------------------------------------------------------------------
        # 以下部分是实体图的推理
        # 需要引入本地上下文池化方法
        # ---------------------------------------------------------------------------------------------------------
        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2] 实体节点从1开始计数改为从0开始
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)

        # [batch_size, h_t_limit, bank_size]
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1)
        # print(entity_graph_feature.shape,entity_graph_attention.shape)
        entity_graphs = params['entity_graphs']
        # entity_graphs = entity_graphs[graph_range1:graph_range2]
        entity_graph_big = dgl.batch(entity_graphs)
        # self.edge_layer(entity_graph_big, entity_graph_feature, entity_graph_attention, context_output, entity_mention_list)
        self.edge_layer(entity_graph_big, entity_graph_feature)

        entity_graphs = dgl.unbatch(entity_graph_big)
        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_out_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table']
        # path_table = path_table[graph_range1:graph_range2]
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(h_t_limit):
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break

                h = h_t_pairs[i, j, 0].item()
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue

                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].number_of_nodes())
                    print(entity_graphs[i].all_edges())
                    print(path_table)
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0:
                    continue

                # forward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v))
                # h 到每个中间节点 v ：[h,v1],[h,v2]...
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')

        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )

        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))
        # predictions = self.predict(torch.cat((h_entity, t_entity, torch.abs(h_entity-t_entity), torch.mul(h_entity, t_entity), global_info), dim=-1))
        return predictions


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layer, output_size):
        super(TransformerEncoder, self).__init__()
        self.mention_max_num = 12#设置实体最多的mention数
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformerencoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.fc = nn.Linear(d_model*self.mention_max_num, output_size)

    def forward(self, x, mask=None):
        #x:[mention_num, bank_size]
        x = x.unsqueeze(0)
        x = self.transformerencoder(x).squeeze(0)
        x = los_sum_exp(x)
        # if len(x) > self.mention_max_num:
        #     x = x[:self.mention_max_num]
        # else:
        #     t = get_cuda(torch.Tensor(self.mention_max_num - len(x), len(x[0])).fill_(torch.finfo(torch.float32).min))
        #     x = torch.cat([x, t], dim=0)
        # x = self.fc(x.reshape(1, -1))
        return x

class Attention(nn.Module):
    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        """
        src: [src_size]
        trg: [middle_node, trg_size]
        """

        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)

        return score.squeeze(0), value.squeeze(0)


class BiLSTM(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.lstm_hidden_size,
                            num_layers=config.nlayers, batch_first=True,
                            bidirectional=True)
        self.in_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_lengths):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                      padding_value=self.config.word_pad)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)

        src_h_t = src_h_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        src_c_t = src_c_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
        output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)

        return outputs, (output_h_t, output_c_t)


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        # self.conv = dglnn.HeteroGraphConv({
        #     rel: dglnn.GATConv(in_feat, out_feat, num_heads=4, attn_drop=dropout)
        #     for rel in rel_names
        # })

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.aggragate = nn.Linear(4*out_feat,out_feat)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs)

        def _apply(ntype, h):
            # print(h.shape)
            # h = self.aggragate(h.view(-1,4*self.out_feat))
            # print(inputs[ntype].shape,self.loop_weight.shape,h.shape)
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


# class RelEdgeLayer(nn.Module):
#     def __init__(self,
#                  node_feat,
#                  edge_feat,
#                  activation,
#                  dropout=0.0):
#         super(RelEdgeLayer, self).__init__()
#         self.node_feat = node_feat
#         self.edge_feat = edge_feat
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout)
#         self.linear_h = nn.Linear(self.node_feat, self.node_feat, bias=False)
#         self.linear_c = nn.Linear(self.node_feat, self.node_feat, bias=False)
#         self.mapping = nn.Linear(self.node_feat * 2, self.node_feat)
#         self.linear_q = nn.Linear(self.node_feat, self.node_feat, bias=False)
#         self.linear_k = nn.Linear(self.node_feat, self.node_feat, bias=False)
#         self.scaling = node_feat ** -0.5

#     def forward(self, g, inputs, attns, context_output, entity_mention_list):
#         # attns: [num_nodes, num_head, s_len]
#         # context_output: [num_nodes, s_len, node_feat]
#         # entity_mention_list: dict of entity-mention
#         # g = g.local_var()
#         g.ndata['h'] = inputs  # [num_nodes, node_feat]
#         g.ndata['a'] = attns  # [num_nodes, num_head, s_len]
#         g.ndata['d'] = context_output
#         g.apply_edges(entity_cat)
#         attn_rate = g.edata['a'] # [num_edges,s_len]
#         edge_D = g.edata['D'] # [num_edges, s_len, node_feat]
#         attn_rate = attn_rate.unsqueeze(-1).transpose(1, 2)
#         c = torch.matmul(attn_rate, edge_D).squeeze(1) # [num_edges, node_feat]
#         c_linear = self.linear_q(c)
#         mention_list = {i:self.linear_k(entity_mention_list[i]) for i in entity_mention_list.keys()}
#         z_s = self.linear_h(g.edata['h_s']) + self.linear_c(c)
#         z_o = self.linear_h(g.edata['h_o']) + self.linear_c(c)
#         g.edata['h'] = self.dropout(self.activation(self.mapping(torch.cat((z_s,z_o),dim=-1))))
#         g.ndata.pop('h')
#         g.ndata.pop('a')
#         g.ndata.pop('d')

class RelEdgeLayer(nn.Module):
    def __init__(self,
                 node_feat,
                 edge_feat,
                 activation,
                 dropout=0.0):
        super(RelEdgeLayer, self).__init__()
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mapping = nn.Linear(node_feat * 2, edge_feat)

    def forward(self, g, inputs):
        # g = g.local_var()

        g.ndata['h'] = inputs  # [total_mention_num, node_feat]
        g.apply_edges(lambda edges: {
            'h': self.dropout(self.activation(self.mapping(torch.cat((edges.src['h'], edges.dst['h']), dim=-1))))})
        g.ndata.pop('h')


class Bert:
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_class, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = 512

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        # assert ids.size(1) < self.max_len
        ids = ids[:, :self.max_len]  # https://github.com/DreamInvoker/GAIN/issues/4
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 512
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        return subword_ids.numpy(), token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])
