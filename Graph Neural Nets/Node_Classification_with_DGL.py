"""

GNN for Classification

"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


import dgl.data

dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)


g = dataset[0]

print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)