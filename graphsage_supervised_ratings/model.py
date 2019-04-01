import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import os
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage_supervised.encoders import Encoder
from graphsage_supervised.aggregators import MeanAggregator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
CLASS_NUM = 2
LIKE_THRESHOLD = 3


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        # self.xent = nn.BCELoss(reduction='mean')  # reduction='none'
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim * 2))
        init.xavier_uniform_(self.weight)

    def forward(self, movie_nodes, user_nodes):
        # in the paper, no fc layer in the end
        movie_embeds = self.enc(movie_nodes)
        user_embeds = self.enc(user_nodes)
        merge_movie_user = torch.cat((movie_embeds, user_embeds), 0)
        scores = self.weight.mm(merge_movie_user)
        return scores.t()

    def loss(self, movie_nodes, user_nodes, labels):
        scores = self.forward(movie_nodes, user_nodes)
        # print(scores.shape)
        # print(labels.squeeze().shape)
        # return self.xent(torch.sigmoid(scores.cuda()), torch.FloatTensor(labels.squeeze()).cuda())
        print(scores)
        return self.xent(scores.cuda(), torch.from_numpy(labels).cuda())

def load_movielens():
    # {node : (nodes in the neighbor)}
    adj_list = defaultdict(set)
    # {(user, movie) : rating}
    rating_list = defaultdict()
    # the dataset contains 943 users and 1682 movies
    num_nodes = 943+1682

    with open(os.path.abspath('../ml-100k/u.data'), 'r') as u_f:
        for i, each_pair in enumerate(u_f.readlines()):
            pair = each_pair.rstrip('\n').split('\t')
            user = int(pair[0]) - 1
            movie = int(pair[1]) - 1
            rating = int(pair[2])
            # rating_list[(user+1682, movie)] = [1, 0] if rating >= LIKE_THRESHOLD else [0, 1]
            rating_list[(user + 1682, movie)] = 1 if rating >= LIKE_THRESHOLD else 0
            # if user not in user_map.keys():
            #     user_map[user] = len(user_map)
            # if movie not in movie_map.keys():
            #     movie_map[movie] = len(movie_map) + 943
            # Users and items are numbered consecutively from 1.
            adj_list[user+1682].add(movie)
            adj_list[movie].add(user+1682)
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
    # feat_data = np.random.rand(num_nodes, num_nodes)
    feat_data = np.eye(num_nodes)
    # print(rating_list)
    return feat_data, rating_list, adj_list

def run_movielens():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2625
    movie_nodes = 1682
    feat_data, rating_list, adj_list = load_movielens()
    # print(labels)
    features = nn.Embedding(num_nodes, num_nodes)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, num_nodes, 100, adj_list, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list, agg2,
                   base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(CLASS_NUM, enc2)
    graphsage.cuda()
    rand_indices = np.random.permutation(movie_nodes)
    # test = rand_indices[:10]
    # val = rand_indices[10:100]
    train = list(rand_indices[:1682])  # train list contains the movie nodes
    # train2 = train.copy()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.1, momentum=0.9)
    # times = []
    for batch in range(100000):
        batch_nodes = train[:1]  # movie node
        random.shuffle(train)
        # start_time = time.time()
        user_node = random.sample(adj_list[batch_nodes[0]], 1) # user node

        optimizer.zero_grad()
        new_lables = [rating_list[(user_node[0], batch_nodes[0])]]
        # for i in batch_nodes:
        #     new_lables.append(labels[i])
        loss = graphsage.loss(batch_nodes, user_node,
                              np.array(new_lables))
                            # labels[[np.array([int(i-1) for i in batch_nodes])]]
                              # labels[np.array([int(i-1) for i in batch_nodes])])
        # print(loss.sum().shape)
        loss.backward()
        optimizer.step()
        # end_time = time.time()
        # times.append(end_time - start_time)
        print(batch, loss.data)

    # val_output = graphsage.forward(train2)

    # for i in val_output:
    #     print(i)
    # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    # run_cora()
    run_movielens()
    # load_movielens()