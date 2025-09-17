import torch
import torch.nn as nn
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss, InfoNCE, InfoNCE_weight
import time
from util.algorithm import find_k_largest
from os.path import abspath
import sys
from util.metrics import ranking_evaluation
from util.FileIO import FileIO
from util.logger import Log
import numpy as np
import random
import scipy.sparse as sp
from datetime import datetime
import matplotlib.pyplot as plt
from util.pagerank import build_graph, PageRank, normalize
import math
import os
from sklearn.manifold import TSNE

class SGL():
    def __init__(self, args, data):
        print("Recommender: SGL")
        self.data = data
        self.args = args
        self.bestPerformance = []
        self.recOutput = []
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)

        # Hyperparameter
        # SGL=-n_layer 2 -lambda 0.1 -droprate 0.1 -augtype 2 -temp 0.2
        self.n_layers = 2
        self.cl_rate = 0.2
        self.aug_type = 2
        self.drop_rate = 0.1
        self.temp = 0.2
        self.model = SGL_Encoder(self.data, self.args.emb_size, self.drop_rate, self.n_layers, self.temp, self.aug_type)
        self.weight_p = None
        self.user_real_p = None
        self.item_real_p = None

    def generate_anchor(self, sorted_list, attackModelName, anchor_size):
        top_5_percent_list = x[:math.ceil(len(sorted_list) * anchor_size)]
        anchor_data = sp.csr_matrix((self.data.interaction_mat.shape[0], self.data.interaction_mat.shape[1]),dtype=self.data.interaction_mat.dtype)
        
        for row_idx in top_5_percent_list:
            anchor_data[row_idx] = self.data.interaction_mat.getrow(row_idx)
        print("\n")
        print("_"*80)
        print("Anchor Data Generating...")
        print("Anchor data generate by {} has done.".format( attackModelName))
            
        
        print("_"*80)
        print("\n")
        return anchor_data
    
    def build_cl_weight(self, user_real_p, item_real_p):
        user_real_p = list(user_real_p.values())
        item_real_p = list(item_real_p.values())
        user_item_p = user_real_p + item_real_p
        n_nodes = len(user_real_p) + len( item_real_p)
        weight = torch.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                weight[i, j] = self.compute_weight(int(user_item_p[i]), int(user_item_p[j]))
        return weight
    
    def compute_weight(self, p1, p2):
        return p1 * p2 + 10 * abs(p1 - p2)
     
    def build_weight(self):
        n_nodes = self.data.user_num + self.data.item_num
        
        item_data = []
        item_row_indices = []
        item_col_indices = []
        
        for user in range(self.data.user_num):
            for item in self.data.interaction_mat[user].nonzero()[1]:
                item_data.append(self.item_real_p[item])
                item_row_indices.append(user)
                item_col_indices.append(item + self.data.user_num)
        
        item_weight_matrix = sp.csr_matrix((item_data, (item_row_indices, item_col_indices)), shape=(n_nodes, n_nodes))

        user_data = []
        user_row_indices = []
        user_col_indices = []
        
        for item in range(self.data.item_num):
            for user in self.data.interaction_mat[:, item].nonzero()[0]:
                user_data.append(self.user_real_p[user])
                user_row_indices.append(item + self.data.user_num)
                user_col_indices.append(user)
        
        user_weight_matrix = sp.csr_matrix((user_data, (user_row_indices, user_col_indices)), shape=(n_nodes, n_nodes))
        weight_matrix = item_weight_matrix + user_weight_matrix
        
        return weight_matrix
    
    
    def train_weight(self,Epoch=0, optimizer=None, evalNum=5, attackModelName=None):
        
        self.bestPerformance = []
        model = SGL_Encoder(self.data, self.args.emb_size, self.drop_rate, self.n_layers, self.temp, self.aug_type).cuda()

        if optimizer is None: optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
       
        maxEpoch = self.args.maxEpoch
        if Epoch: maxEpoch = Epoch
        for epoch in range(maxEpoch):
           
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            if epoch == 6:
                self.data.loss = {key: value / self.data.interaction_mat[key].nonzero()[1].shape[0] for key, value in self.data.loss.items()}
                sorted_loss = sorted(self.data.loss.items(), key=lambda item: item[1])
                anchor_data = self.generate_anchor(sorted_loss, attackModelName,anchor_size=0.01)
                x = [item[0] for item in sorted_loss]
                y = [item[1] for item in sorted_loss]
                
                G = build_graph(anchor_data)
                item_pagerank = PageRank(G)
                item_fake_p = normalize(item_pagerank)
                self.item_real_p = {key: 1 - value for key, value in item_fake_p.items()}
                
                item_indices = np.unique(self.data.interaction_mat.indices)
                self.user_real_p = {}
                for item in item_indices:
                    if item not in self.item_real_p.keys():
                        self.item_real_p[item] = 1
                # alpha = 0.5
                # for item in item_indices:
                #     if self.item_real_p[item] < alpha:
                #         self.item_real_p[item] = 0
                         
                for user in range(self.data.user_num):
                    interaction_item = self.data.interaction_mat[user].nonzero()[1]
                    real_pro = sum([self.item_real_p[i] for i in interaction_item]) / len(interaction_item)
                    # if real_pro < alpha:
                    #     self.user_real_p[user] = 0
                    # else:
                    #     self.user_real_p[user] = 1
                    self.user_real_p[user] = real_pro
                
                self.weight_p = self.build_weight()  
                self.weight_cl = self.build_cl_weight(self.user_real_p, self.item_real_p)
                break
            
            for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                
                if epoch == 5:
                    for i in range(len(user_idx)):
                        temp_user_emb, temp_pos_item_emb, temp_neg_item_emb = user_emb[i].unsqueeze(0), pos_item_emb[i].unsqueeze(0), neg_item_emb[i].unsqueeze(0)
                        loss = bpr_loss(temp_user_emb, temp_pos_item_emb, temp_neg_item_emb) + l2_reg_loss(self.args.reg, temp_user_emb, temp_pos_item_emb) + self.cl_rate * model.cal_cl_loss([[user_idx[i]], [pos_idx[i]]], dropped_adj1, dropped_adj2)
                        if user_idx[i] not in self.data.loss:
                            self.data.loss[user_idx[i]] = loss.item()
                        else:
                            self.data.loss[user_idx[i]] += loss.item()
                            
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], dropped_adj1, dropped_adj2)
                batch_loss = rec_loss + l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()

                
                optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % evalNum == 0:
                self.evaluate(epoch)
        
        
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    
    def train_cl(self, Epoch=0, optimizer=None,
              evalNum=5):
        self.bestPerformance = []
        model = self.model.cuda()
        if optimizer is None: optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
       
        maxEpoch = self.args.maxEpoch
        if Epoch: maxEpoch = Epoch
        for epoch in range(maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            
            for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                # if any(user in user_idx for user in poison_user):
                #     print("yes")
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss_weight([user_idx, pos_idx], dropped_adj1, dropped_adj2, self.weight_cl)
                batch_loss = rec_loss + l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()

                optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % evalNum == 0:
                self.evaluate(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
           
    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None,
              evalNum=5):
        self.bestPerformance = []
        model = self.model.cuda()
        if optimizer is None: optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        if requires_adjgrad: gradAll = torch.zeros(self.data.user_num, self.data.item_num).cuda()
        if requires_embgrad:
            self.model.requires_grad = True
            self.usergrad = torch.zeros((self.data.user_num, self.args.emb_size)).cuda()
            self.itemgrad = torch.zeros((self.data.item_num, self.args.emb_size)).cuda()
        maxEpoch = self.args.maxEpoch
        if Epoch: maxEpoch = Epoch
        for epoch in range(maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            if requires_adjgrad:
                grad_mat1 = torch.zeros_like(dropped_adj1)
                grad_mat2 = torch.zeros_like(dropped_adj2)
                if isinstance(dropped_adj1, list):
                    for i in range(self.n_layers):
                        dropped_adj1[i].requires_grad = True
                        dropped_adj2[i].requires_grad = True
                else:
                    dropped_adj1.requires_grad = True
                    dropped_adj2.requires_grad = True
            for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], dropped_adj1, dropped_adj2)
                batch_loss = rec_loss + l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()

                if requires_adjgrad:
                    grad_mat1 += dropped_adj1.grad
                    grad_mat2 += dropped_adj2.grad
                optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            if requires_adjgrad and maxEpoch - epoch < gradIterationNum:
                gradAll += (grad_mat1 + grad_mat2).to_dense()[:self.data.user_num, self.data.user_num:]
            elif requires_embgrad and maxEpoch - epoch < gradIterationNum:
                self.usergrad += self.model.embedding_dict["user_emb"].grad
                self.itemgrad += self.model.embedding_dict["item_emb"].grad
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % evalNum == 0:
                self.evaluate(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        
        
        if requires_adjgrad and requires_embgrad:
            return gradAll, self.user_emb, self.item_emb, self.usergrad, self.itemgrad
        elif requires_adjgrad:
            return gradAll
        elif requires_embgrad:
            return self.user_emb, self.item_emb, self.usergrad, self.itemgrad

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def evaluate(self, epoch):
        print('Evaluating the model...')
        rec_list, _ = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list,ranking_evaluation(self.data.test_set, rec_list, self.topN)


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_uiAdj(self, ui_adj):
        self.sparse_norm_adj = sp.diags(np.array((1 / np.sqrt(ui_adj.sum(1)))).flatten()) @ ui_adj @ sp.diags(
            np.array((1 / np.sqrt(ui_adj.sum(0)))).flatten())
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.sparse_norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def attack_emb(self, users_emb_grad, items_emb_grad):
        with torch.no_grad():
            self.embedding_dict['user_emb'] += users_emb_grad
            self.embedding_dict['item_emb'] += items_emb_grad

    def graph_reconstruction(self):
        if self.aug_type == 0 or self.aug_type == 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0).cuda()
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        return InfoNCE(view1, view2, self.temp)
    
    def cal_cl_loss_weight(self, idx, perturbed_mat1, perturbed_mat2, weight):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        idx = u_idx.tolist() + i_idx.tolist()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        weight = weight[idx,:][:, idx].cuda()
        return InfoNCE_weight(view1, view2, self.temp, weight)

   

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


class GraphAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def node_dropout(sp_adj, drop_rate):
        """Input: a sparse adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj
