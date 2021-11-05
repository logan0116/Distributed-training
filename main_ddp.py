import os
import sys
import tempfile
import torch
import json
import numpy as np
import networkx as nx
import torch.distributed as dist
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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
        source_list = []
        target_list = []
        node_ng_list = []

        for link in self.link_list_weighted:
            source_list.append(link[0])
            target_list.append(link[1])
            node_ng = np.random.choice(sample_table, size=(self.ng_num)).tolist()
            node_ng_list.append(node_ng)

        return self.node_num, \
               torch.LongTensor(source_list), torch.LongTensor(target_list), torch.LongTensor(node_ng_list)


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
    def forward(self, s, t, ng):
        vector_i = self.u_emd(s)
        # 一阶
        vector_o1 = self.u_emd(t)
        vector_ng1 = self.u_emd(ng)
        output_1_1 = torch.matmul(vector_i, vector_o1.transpose(-1, -2)).squeeze()
        output_1_1 = F.logsigmoid(output_1_1)
        # 负采样的部分
        output_1_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng1.transpose(-1, -2)).squeeze()
        output_1_2 = F.logsigmoid(-1 * output_1_2).sum(1)

        output_1 = -1 * (output_1_1 + output_1_2)
        # 二阶
        vector_o2 = self.context_emd(t)
        vector_ng2 = self.context_emd(ng)
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

    def __init__(self, s_list, t_list, ng_list):
        self.s_list = s_list
        self.t_list = t_list

        self.ng_list = ng_list

    def __len__(self):
        return len(self.s_list)

    def __getitem__(self, idx):
        return self.s_list[idx], self.t_list[idx], self.ng_list[idx]


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # 参数设置
    d = 768
    batch_size = 8
    epochs = 35
    ng_num = 5

    node_path = '../data/input/node.json'
    link_path = '../data/input/link.json'

    # 数据处理
    networkdeal = NetworkDeal(node_path, link_path, ng_num)
    networkdeal.get_network_feature()
    node_size, s_list, t_list, ng_list = networkdeal.get_data()
    loader = Data.DataLoader(MyDataSet(s_list, t_list, ng_list), batch_size, True)

    # create model and move it to GPU with id rank
    model = Line(node_size, d).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.0001)
    optimizer.zero_grad()

    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        for i, (s, t, ng) in enumerate(loader):
            s = s.to(rank)
            t = t.to(rank)
            ng = ng.to(rank)
            loss = ddp_model(s, t, ng)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print("epoch", epoch, "loss", loss.item())

    cleanup()
    print('train_time:', time.time() - start_time)


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus = 1
    run_demo(demo_basic, world_size)
