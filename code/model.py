"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import random


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class SRGNN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(SRGNN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.dropout = nn.Dropout(0.1)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)


    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.dropout = nn.Dropout(0.5)
        self.ssl_ratio = 0.2
        self.beta = 0.001
        self.T = 0.1
        # 计算要选择的节点数量
        num_users = int(self.num_users * self.ssl_ratio)
        num_items = int(self.num_items * self.ssl_ratio)

        # 生成节点ID列表
        users_ids = list(range(self.num_users))
        items_ids = list(range(self.num_items))

        # 随机选择节点ID
        drop_user_idx = random.sample(users_ids, num_users)
        drop_item_idx = random.sample(items_ids, num_items)
        user_mask = torch.ones((self.num_users, self.latent_dim))
        item_mask = torch.ones((self.num_items, self.latent_dim))
        user_mask[drop_user_idx] = 0.
        self.user_mask = user_mask.to(world.device)
        item_mask[drop_item_idx] = 0.
        self.item_mask = item_mask.to(world.device)
        #print(drop_item_idx)
        #drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
        #drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        """self.embedding_user.weight.data *= user_mask
        self.embedding_item.weight.data *= item_mask
        self.embedding_user = self.embedding_user.weight.data
        self.embedding_item = self.embedding_item.weight.data"""


        self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=1, batch_first=True)
        self.batch = 1

        initializer = nn.init.xavier_uniform_

        weight_dict = nn.ParameterDict()
        for k in range(self.n_layers):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(self.latent_dim,
                                                                                    self.latent_dim)))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, self.latent_dim)))})

            """weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(self.latent_dim,
                                                                                    self.latent_dim)))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, self.latent_dim)))})"""

            weight_dict.update({'h_%d' % k: nn.Parameter(
                initializer(torch.empty(1, int((self.num_users + self.num_items) / self.batch), self.latent_dim)))})
        self.weight_dict = weight_dict

        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
       

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer1(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, embs

    def computer2(self):
        """
        propagate methods for srgnn
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        """self.embedding_user.weight.data *= self.user_mask
        self.embedding_item.weight.data *= self.item_mask
        users_emb = self.embedding_user.weight.data
        items_emb = self.embedding_item.weight.data"""

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for k in range(self.n_layers):
            num_batches = all_emb.shape[0] // self.batch
            nodes = all_emb[:num_batches * self.batch, :]
            nodes = nodes.reshape((self.batch, num_batches, self.latent_dim)).transpose(0, 1)
            # output, h = self.rnn(nodes, h)
            output, h = self.gru(nodes, self.weight_dict['h_%d' % k])  # (70839,64)
            side_emb = output.transpose(0, 1).reshape((num_batches * self.batch, self.latent_dim))

            # side_emb = self.gru(side_emb)  # (70839,64)

            side_emb = torch.mul(all_emb, side_emb)

            side_emb = torch.matmul(side_emb, self.weight_dict['W_gc_%d' % k]) \
                       + self.weight_dict['b_gc_%d' % k]
            side_emb = nn.LeakyReLU(negative_slope=0.2)(side_emb + all_emb)
            side_emb = self.dropout(side_emb)
            all_emb = F.normalize(side_emb, p=2, dim=1)

            all_emb = torch.sparse.mm(g_droped, all_emb)

            # all_emb = (1/(k+2)) * all_emb
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.sum(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, embs
    
    def getUsersRating(self, users):
        all_users1, all_items1, _ = self.computer1()
        all_users2, all_items2, _ = self.computer2()
        users_emb1 = all_users1[users.long()]
        users_emb2 = all_users2[users.long()]
        users_emb = users_emb1 + self.beta * users_emb2
        items_emb1 = all_items1
        items_emb2 = all_items2
        items_emb = items_emb1 + self.beta * items_emb2
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding1(self, users, pos_items, neg_items):
        all_users, all_items, _ = self.computer1()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getEmbedding2(self, users, pos_items, neg_items):
        all_users, all_items, _ = self.computer2()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getEmbedding3(self,):
        _, _, embs_list1 = self.computer1()
        _, _, embs_list2 = self.computer2()
        #embedding_lgc = torch.mean(embs_list1, dim=1)
        #embedding_kgc = torch.sum(embs_list2, dim=1)

        center_embedding1 = embs_list1[:,0, :]
        center_embedding2 = embs_list2[:,0, :]
        center_total_embedding = center_embedding1 + center_embedding2
        center_local_embedding = center_embedding2

        context_embedding1 = embs_list1[:,2, :]
        context_embedding2 = embs_list2[:,2, :]
        context_total_embedding = context_embedding1 + context_embedding2
        context_local_embedding = context_embedding2


        """users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)"""
        return context_total_embedding, center_total_embedding, center_local_embedding, context_local_embedding

    def bpr_loss1(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding1(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    def bpr_loss2(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding2(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def cl_loss(self, context_total_embedding, center_total_embedding, embedding_lgc, embedding_kgc, users, items):
        # 全局loss
        context_user_embeddings, context_item_embeddings = torch.split(context_total_embedding,
                                                                       [self.num_users, self.num_items])
        center_user_embeddings_all, center_item_embeddings_all = torch.split(center_total_embedding,
                                                                             [self.num_users, self.num_items])

        context_user_embeddings = context_user_embeddings[users]
        center_user_embeddings = center_user_embeddings_all[users]
        norm_user_emb1 = F.normalize(context_user_embeddings)
        norm_user_emb2 = F.normalize(center_user_embeddings)
        norm_all_user_emb = F.normalize(center_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.T)
        ttl_score_user = torch.exp(ttl_score_user / self.T).sum(dim=1)
        cl_loss_user_total = -torch.log(pos_score_user / ttl_score_user).sum()
        # ssl = ssl_loss_user
        context_item_embeddings = context_item_embeddings[items]
        center_item_embeddings = center_item_embeddings_all[items]
        norm_item_emb1 = F.normalize(context_item_embeddings)
        norm_item_emb2 = F.normalize(center_item_embeddings)
        norm_all_item_emb = F.normalize(center_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.T)
        ttl_score_item = torch.exp(ttl_score_item / self.T).sum(dim=1)
        cl_loss_item_total = -torch.log(pos_score_item / ttl_score_item).sum()

        #  局部loss
        """context_user_embeddings_local, context_item_embeddings_local = torch.split(context_local_embedding,
                                                                       [self.num_users, self.num_items])
        center_user_embeddings_all_local, center_item_embeddings_all_local = torch.split(center_local_embedding,
                                                                             [self.num_users, self.num_items])

        context_user_embeddings_local = context_user_embeddings_local[users]
        center_user_embeddings_local = center_user_embeddings_all_local[users]
        norm_user_emb1 = F.normalize(context_user_embeddings_local)
        norm_user_emb2 = F.normalize(center_user_embeddings_local)
        norm_all_user_emb = F.normalize(center_user_embeddings_all_local)
        pos_score_user_local = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user_local = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user_local = torch.exp(pos_score_user_local / 0.1)
        ttl_score_user_local = torch.exp(ttl_score_user_local / 0.1).sum(dim=1)
        cl_loss_user_local = -torch.log(pos_score_user_local / ttl_score_user_local).sum()
        # ssl = ssl_loss_user
        context_item_embeddings_local = context_item_embeddings_local[items]
        center_item_embeddings_local = center_item_embeddings_all_local[items]
        norm_item_emb1 = F.normalize(context_item_embeddings_local)
        norm_item_emb2 = F.normalize(center_item_embeddings_local)
        norm_all_item_emb = F.normalize(center_item_embeddings_all)
        pos_score_item_local = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item_local = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item_local = torch.exp(pos_score_item_local / 0.1)
        ttl_score_item_local = torch.exp(ttl_score_item_local / 0.1).sum(dim=1)
        cl_loss_item_local = -torch.log(pos_score_item_local / ttl_score_item_local).sum()"""
        embedding_lgc_user, embedding_lgc_item = torch.split(embedding_lgc, [self.num_users, self.num_items])
        embedding_kgc_user, embedding_kgc_item = torch.split(embedding_kgc, [self.num_users, self.num_items])

        context_user_embeddings1 = embedding_kgc_user[users]
        center_user_embeddings1 = embedding_lgc_user[users]
        norm_user_emb3 = F.normalize(context_user_embeddings1)
        norm_user_emb4 = F.normalize(center_user_embeddings1)
        norm_all_user_emb1 = F.normalize(embedding_kgc_user)
        pos_score_user1 = torch.mul(norm_user_emb3, norm_user_emb4).sum(dim=1)
        ttl_score_user1 = torch.matmul(norm_user_emb3, norm_all_user_emb1.transpose(0, 1))
        pos_score_user1 = torch.exp(pos_score_user1 / self.T)
        ttl_score_user1 = torch.exp(ttl_score_user1 / self.T).sum(dim=1)
        cl_loss_user_local = -torch.log(pos_score_user1 / ttl_score_user1).sum()
        # ssl = ssl_loss_user
        context_item_embeddings2 = embedding_kgc_item[items]
        center_item_embeddings2 = embedding_lgc_item[items]
        norm_item_emb5 = F.normalize(context_item_embeddings2)
        norm_item_emb6 = F.normalize(center_item_embeddings2)
        norm_all_item_emb1 = F.normalize(embedding_kgc_item)
        pos_score_item2 = torch.mul(norm_item_emb5, norm_item_emb6).sum(dim=1)
        ttl_score_item2 = torch.matmul(norm_item_emb5, norm_all_item_emb1.transpose(0, 1))
        pos_score_item2 = torch.exp(pos_score_item2 / self.T)
        ttl_score_item2 = torch.exp(ttl_score_item2 / self.T).sum(dim=1)
        cl_loss_item_local = -torch.log(pos_score_item2 / ttl_score_item2).sum()


        return cl_loss_user_total, cl_loss_item_total, cl_loss_user_local, cl_loss_item_local
       
    def forward(self, users, items):
        # compute embedding
        all_users1, all_items1 = self.computer1()
        all_users2, all_items2 = self.computer2()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb1 = all_users1[users]
        items_emb1 = all_items1[items]
        users_emb2 = all_users2[users]
        items_emb2 = all_items2[items]

        users_emb = users_emb1 + self.beta * users_emb2
        items_emb = items_emb1 + self.beta * items_emb2
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
