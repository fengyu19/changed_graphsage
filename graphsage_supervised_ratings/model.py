import torch
import torch.nn as nn
from torch.nn import init

import os
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import pickle

from graphsage_supervised.encoders import Encoder
from graphsage_supervised.aggregators import MeanAggregator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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
        self.m = nn.Softmax()

    def forward(self, movie_nodes, user_nodes):
        movie_embeds = self.enc(movie_nodes)
        user_embeds = self.enc(user_nodes)
        merge_movie_user = torch.cat((movie_embeds, user_embeds), 0)
        scores = self.weight.mm(merge_movie_user)
        scores = scores.t()
        # print(self.m(scores.data))
        return movie_embeds, user_embeds, scores

    def loss(self, movie_nodes, user_nodes, labels):
        movie_embeds, user_embeds, scores = self.forward(movie_nodes, user_nodes)
        return self.xent(scores.cuda(), torch.from_numpy(labels).cuda())
        # return self.xent(torch.sigmoid(scores.cuda()), torch.FloatTensor(labels.squeeze()).cuda())

    def embeddings(self, node_id):
        return self.enc(node_id)


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
            adj_list[user+1682].add(movie)
            adj_list[movie].add(user+1682)
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
    train = list(rand_indices[:movie_nodes])  # train list contains the movie nodes
    # train2 = train.copy()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.001, momentum=0.9)
    # times = []
    for batch in range(10):
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
        loss.backward()
        optimizer.step()
        # end_time = time.time()
        # times.append(end_time - start_time)
        # print(batch, loss.data)
    torch.save(graphsage.state_dict(), "../models/graphsage_ratings_model")
    # val_output = graphsage.forward(train2)

    # for i in val_output:
    #     print(i)
    # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Average batch time:", np.mean(times))

def save_embedings():
    num_nodes = 2625
    movie_nodes = 1682
    user_nodes = 943
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

    dict_result = defaultdict()
    temp_result = defaultdict()
    graphsage = SupervisedGraphSage(CLASS_NUM, enc2)
    graphsage.cuda()
    graphsage.load_state_dict(torch.load("../models/graphsage_ratings_model"))
    for user_id in range(movie_nodes+1, user_nodes+movie_nodes+1):
        for movie_id in range(1, movie_nodes+1):
            movie_embeds, user_embeds, scores = graphsage.forward([movie_id], [user_id])
            temp_result[movie_id] = scores.cpu().detach().numpy()
            print(user_id, movie_id)
        dict_result[user_id] = temp_result
        temp_result = defaultdict()
    output = open('scores.pkl', 'wb')
    pickle.dump(dict_result, output)

    # for node_id in range(1, num_nodes+1):
    #     embedding = graphsage.embeddings([node_id])
    #     embedding = embedding.cpu().detach().numpy()
    #     np.save("../embedding_results/{}.npy".format(str(node_id)), embedding)
        # print(node_id, embedding)


if __name__ == "__main__":
    # run_movielens()
    save_embedings()
    # load_movielens()