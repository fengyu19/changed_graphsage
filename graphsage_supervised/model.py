import torch
import torch.nn as nn
from torch.nn import init

import os
import numpy as np
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage_supervised.encoders import Encoder
from graphsage_supervised.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.BCELoss(reduction='mean')  # reduction='none'

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        # in the paper there is no fc layer in the end
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        # print(scores.shape)
        # print(labels.squeeze().shape)
        return self.xent(torch.sigmoid(scores.cuda()), torch.FloatTensor(labels.squeeze()).cuda())


def load_MovieLens():
    # movie2user = defaultdict(set)
    # {node : {nodes in the neighbor}}
    adj_lists = defaultdict(set)
    user_map = {}
    movie_map = {}

    # the dataset contains 943 users and 1682 movies
    num_nodes = 943+1682

    with open(os.path.abspath('../ml-100k/u.data'), 'r') as u_f:
        for i, each_pair in enumerate(u_f.readlines()):
            pair = each_pair.rstrip('\n').split('\t')
            user = int(pair[0]) - 1
            movie = int(pair[1]) - 1
            rating = int(pair[2])

            # if user not in user_map.keys():
            #     user_map[user] = len(user_map)
            # if movie not in movie_map.keys():
            #     movie_map[movie] = len(movie_map) + 943
            # Users and items are numbered consecutively from 1.
            adj_lists[user+1682].add(movie)
            adj_lists[movie].add(user+1682)
            # movie2user[movie_map[movie]].add(user_map[user])
        # print(adj_lists[943])
        # print(movie_map)

    # network: user2user the neighbors of movies are still movies
    # for i in movie2user.keys():
    #     for j in movie2user.keys():
    #         if i != j:
    #             for i_item in movie2user[i]:
    #                 if i_item in movie2user[j]:
    #                     adj_lists[i].add(j)
    #                     break
    # for i in range(len(adj_lists)):
    #     print(len(adj_lists[i]))

    # feat_data = np.identity(len(adj_lists.keys()))
    with open(os.path.abspath('../ml-100k/u.item'), 'r') as i_f:
        labels = []
        for i, each_line in enumerate(i_f.readlines()):
            label = each_line.rstrip('\n')[-37:].split('|')
            labels.append([float(i) for i in label])

    feat_data = np.eye(num_nodes)

    return feat_data, labels, adj_lists

def run_MovieLens():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2625
    movie_nodes = 1682
    feat_data, labels, adj_lists = load_MovieLens()
    # print(labels)
    features = nn.Embedding(num_nodes, num_nodes)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 2625, 100, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 10


    graphsage = SupervisedGraphSage(19, enc2)
    graphsage.cuda()
    rand_indices = np.random.permutation(movie_nodes)
    test = rand_indices[:10]
    val = rand_indices[10:100]
    train = list(rand_indices[:1682])
    train2 = train.copy()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    # times = []
    for batch in range(5000):
        batch_nodes = train[:100]
        random.shuffle(train)
        # start_time = time.time()
        optimizer.zero_grad()
        new_lables = []
        for i in batch_nodes:
            new_lables.append(labels[i])
        loss = graphsage.loss(batch_nodes,
                              np.array(new_lables))
                            # labels[[np.array([int(i-1) for i in batch_nodes])]]
                              # labels[np.array([int(i-1) for i in batch_nodes])])
        # print(loss.sum().shape)
        loss.backward()
        optimizer.step()
        # end_time = time.time()
        # times.append(end_time - start_time)
        print(batch, loss.data)

    val_output = graphsage.forward(train2)

    # for i in val_output:
    #     print(i)
    # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    # run_cora()
    run_MovieLens()