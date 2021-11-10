#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 下午3:56
# @Author  : liu yuhan
# @FileName: 2.network_embedding.py
# @Software: PyCharm

# 这个地方还是要考虑两种合并的方案

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import networkx as nx
import json
import time
from tqdm import tqdm


def not_empty(s):
    return s and s.strip()


class NetworkDeal():
    '''
    生成网络，
    '''

    def __init__(self, node_path, link_path, ng_num):
        self.ng_num = ng_num
        with open(node_path, 'r', encoding='UTF-8')as file:
            self.node_list = json.load(file)
        self.node_num = len(self.node_list)
        with open(link_path, 'r', encoding='UTF-8')as file:
            self.link_list_weighted = json.load(file)
        print('node_num:', self.node_num)
        print('link_num:', len(self.link_list_weighted))

    def get_network_feature(self):
        '''
        构建网络计算个重要参数——degree
        :return:
        '''
        g = nx.Graph()
        g.add_nodes_from([i for i in range(self.node_num)])
        g.add_weighted_edges_from(self.link_list_weighted)
        self.node_degree_dict = dict(nx.degree(g))
        return self.node_num

    def get_data(self):
        '''
        负采样
        :return:
        '''
        # 负采样
        sample_table = []
        sample_table_size = 1e8
        # 词频0.75次方
        pow_frequency = np.array(list(self.node_degree_dict.values())) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            sample_table += [wid] * int(c)
        sample_table = np.array(sample_table)

        # 生成一个训练集：
        dataset = []

        for link in self.link_list_weighted:
            node_ng = np.random.choice(sample_table, size=(self.ng_num)).tolist()
            dataset.append(link[:2] + node_ng)
        return torch.LongTensor(dataset)


class Line(nn.Module):
    def __init__(self, word_size, dim):
        super(Line, self).__init__()
        initrange = 0.5 / dim
        # input
        self.u_emd = nn.Embedding(word_size, dim)
        self.u_emd.weight.data.uniform_(-initrange, initrange)
        # output
        self.context_emd = nn.Embedding(word_size, dim)
        self.context_emd.weight.data.uniform_(-0, 0)

    # 这边进行一个一阶+二阶的
    def forward(self, data):
        vector_i = self.u_emd(data[:, 0])
        # 一阶
        vector_o1 = self.u_emd(data[:, 1])
        vector_ng1 = self.u_emd(data[:, 2:])

        output_1_1 = torch.matmul(vector_i, vector_o1.transpose(-1, -2)).squeeze()
        output_1_1 = F.logsigmoid(output_1_1)
        # 负采样的部分
        output_1_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng1.transpose(-1, -2)).squeeze()
        output_1_2 = F.logsigmoid(-1 * output_1_2).sum(1)

        output_1 = -1 * (output_1_1 + output_1_2)
        # 二阶
        vector_o2 = self.context_emd(data[:, 1])
        vector_ng2 = self.context_emd(data[:, 2:])
        output_2_1 = torch.matmul(vector_i, vector_o2.transpose(-1, -2)).squeeze()
        output_2_1 = F.logsigmoid(output_2_1)
        # 负采样的部分
        output_2_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng2.transpose(-1, -2)).squeeze()
        output_2_2 = F.logsigmoid(-1 * output_2_2).sum(1)
        # 组合
        output_2 = -1 * (output_2_1 + output_2_2)
        loss = torch.mean(output_1) + torch.mean(output_2)
        return loss

    # 保存参数
    def save_embedding(self, file_name):
        """
        Save all embeddings to file.
        """
        embedding = self.u_emd.weight.cpu().data.numpy()
        np.save(file_name, embedding)


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, s_list):
        self.s_list = s_list

    def __len__(self):
        return len(self.s_list)

    def __getitem__(self, idx):
        return self.s_list[idx]


if __name__ == '__main__':
    # 参数设置
    d = 768
    batch_size = 8
    epochs = 35

    node_path = '../data/input/node.json'
    link_path = '../data/input/link.json'

    ng_num = 5
    # 数据处理
    networkdeal = NetworkDeal(node_path, link_path, ng_num)
    node_size = networkdeal.get_network_feature()
    train_dataset = networkdeal.get_data()
    loader = Data.DataLoader(MyDataSet(train_dataset), batch_size, True)

    # 模型初始化
    line = Line(node_size, d)
    line.cuda()
    optimizer = optim.Adam(line.parameters(), lr=0.0001)

    # 保存一次参数，作为对比
    # line.save_embedding(node_emb_path_1)
    # 保存平均的loss
    ave_loss = []

    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        loss_collector = []
        for i, data in enumerate(loader):
            data = data.cuda()
            loss = line(data)
            loss.backward()
            optimizer.step()

            # if i % 500 == 0:
            #     print("epoch", epoch, "loss", loss.item())
            loss_collector.append(loss.item())
        ave_loss.append(np.mean(loss_collector))
    print('train_time:', time.time() - start_time)

    # loss_draw(epochs, ave_loss)

    # 保存一次参数，作为对比
    # line.save_embedding(node_emb_path_2)
