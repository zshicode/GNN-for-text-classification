from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
import numpy as np

from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP

import argparse

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Use CUDA training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["GCN", "SAGE", "GAT"],
                        help='model to use.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='require early stopping.')
    parser.add_argument('--dataset', type=str, default='mr',
                        choices = ['20ng', 'R8', 'R52', 'ohsumed', 'mr'],
                        help='dataset to train')

    args, _ = parser.parse_known_args()
    #args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

args = get_citation_args()

# if len(sys.argv) != 2:
# 	sys.exit("Use: python train.py <dataset>")


#dataset = sys.argv[1]

# Set random seed
# seed = random.randint(1, 200)
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda:0')

# Load data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus('mr')
features = sp.identity(features.shape[0])
features = preprocess_features(features)

def pre_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

adjdense = torch.from_numpy(pre_adj(adj).A.astype(np.float32))

def construct_graph(adjacency):
    g = DGLGraph()
    adj = pre_adj(adjacency)
    g.add_nodes(adj.shape[0])
    g.add_edges(adj.row,adj.col)
    adjdense = adj.A
    adjd = np.ones((adj.shape[0]))
    for i in range(adj.shape[0]):
        adjd[i] = adjd[i] * np.sum(adjdense[i,:])
    weight = torch.from_numpy(adj.data.astype(np.float32))
    g.ndata['d'] = torch.from_numpy(adjd.astype(np.float32))
    g.edata['w'] = weight

    if args.cuda:
        g.to(torch.device('cuda:0'))
    
    return g

class SimpleConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super(SimpleConv, self).__init__()
        self.graph = g
        self.activation = activation
        #self.reset_parameters()
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        #self.b = nn.Parameter(torch.zeros(1, out_feats))
        #self.linear = nn.Linear(in_feats,out_feats)
        self.feat_drop = feat_drop
    
    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain('relu')
    #     nn.init.xavier_uniform_(self.linear.weight,gain=gain)
    
    def forward(self, feat):
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.update_all(fn.src_mul_edge(src='h', edge='w', out='m'), fn.sum(msg='m',out='h'))
        rst = g.ndata['h']
        #rst = self.linear(rst)
        rst = self.activation(rst)
        return rst

class SAGEMeanConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation):
        super(SAGEMeanConv, self).__init__()
        self.graph = g
        self.feat_drop = nn.Dropout(0.5)
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        #self.linear = nn.Linear(in_feats, out_feats, bias=True)
        setattr(self, 'Wn', nn.Parameter(torch.randn(out_feats,out_feats)))
        self.activation = activation
        #self.neigh_linear = nn.Linear(out_feats, out_feats, bias=True)
        # self.reset_parameters()
    
    '''
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight,gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight,gain=gain)
    '''
    
    def forward(self,feat):
        g = self.graph.local_var()
        #feat = self.feat_drop(feat)
        h_self = feat.mm(getattr(self, 'W'))
        g.ndata['h'] = h_self
        g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
        h_neigh = g.ndata['neigh']
        degs = g.in_degrees().float()
        degs = degs.to(torch.device('cuda:0'))
        g.ndata['h'] = (h_neigh + h_self) / (degs.unsqueeze(-1) + 1)
        rst = g.ndata['h']
        rst = self.activation(rst)
        # rst = th.norm(rst)
        return rst

class GATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.graph = g
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        setattr(self, 'al', nn.Parameter(torch.randn(in_feats,1)))
        setattr(self, 'ar', nn.Parameter(torch.randn(in_feats,1)))

    def forward(self, feat):
        # equation (1)
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.ndata['el'] = feat.mm(getattr(self, 'al'))
        g.ndata['er'] = feat.mm(getattr(self, 'ar'))
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # message passing
        g.update_all(fn.src_mul_edge('h', 'w', 'm'), fn.sum('m', 'h'))
        e = F.leaky_relu(g.edata['e'])
        # compute softmax
        g.edata['w'] = F.softmax(e)
        rst = g.ndata['h']
        #rst = self.linear(rst)
        #rst = self.activation(rst)
        return rst

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, activation, num_heads=2, merge=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
        self.activation = activation

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            x = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            x = torch.mean(torch.stack(head_outs),dim=0)
        
        return self.activation(x)

class MultiLayer(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super(MultiLayer, self).__init__()
        self.graph = g
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.reset_parameters()
        self.feat_drop = feat_drop
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight,gain=gain)

    def forward(self,feat):
        g = self.graph.local_var()
        if self.feat_drop:
            drop = nn.Dropout(0.5)
            feat = drop(feat)

        rst = self.linear(feat)
        rst = self.activation(rst)
        return rst

class Classifer(nn.Module):
    def __init__(self,g,input_dim,num_classes,conv):
        super(Classifer, self).__init__()
        self.GCN = conv
        self.gcn1 = self.GCN(g,input_dim, 200, F.relu)
        self.gcn2 = self.GCN(g, 200, num_classes, F.relu)
    
    def forward(self, features):
        x = self.gcn1(features)
        self.embedding = x
        x = self.gcn2(x)
        
        return x

g = construct_graph(adj)

# Define placeholders
t_features = torch.from_numpy(features.astype(np.float32))
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
support = [preprocess_adj(adj)]
num_supports = 1
t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

if args.model == 'GCN':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=SimpleConv)
elif args.model == 'SAGE':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=SAGEMeanConv)
elif args.model == 'GAT':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=MultiHeadGATLayer)
else:
    raise NotImplemented
# support has only one element, support[0] is adjacency
if args.cuda and torch.cuda.is_available():
    t_features = t_features.cuda()
    t_y_train = t_y_train.cuda()
    #t_y_val = t_y_val.cuda()
    #t_y_test = t_y_test.cuda()
    t_train_mask = t_train_mask.cuda()
    tm_train_mask = tm_train_mask.cuda()
    # for i in range(len(support)):
    #     t_support = [t.cuda() for t in t_support if True]
    model = model.cuda()

print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def evaluate(features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features).cpu()
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)

val_losses = []

# Train model
for epoch in range(args.epochs):

    t = time.time()
    
    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
        
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > args.early_stopping and val_losses[-1] > np.mean(val_losses[-(args.early_stopping+1):-1]):
        print_log("Early stopping...")
        break


print_log("Optimization Finished!")


# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))


print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


# doc and word embeddings
tmp = model.embedding.cpu().numpy()
word_embeddings = tmp[train_size: adj.shape[0] - test_size]
train_doc_embeddings = tmp[:train_size]  # include val docs
test_doc_embeddings = tmp[adj.shape[0] - test_size:]

print_log('Embeddings:')
print_log('\rWord_embeddings:'+str(len(word_embeddings)))
print_log('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings))) 
print_log('\rTest_doc_embeddings:'+str(len(test_doc_embeddings))) 
print_log('\rWord_embeddings:') 
print(word_embeddings)

with open('./data/corpus/' + args.dataset + '_vocab.txt', 'r') as f:
    words = f.readlines()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
with open('./data/' + args.dataset + '_word_vectors.txt', 'w') as f:
    f.write(word_embeddings_str)

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
with open('./data/' + args.dataset + '_doc_vectors.txt', 'w') as f:
    f.write(doc_embeddings_str)

