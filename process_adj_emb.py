# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle
import os
from sklearn.preprocessing import StandardScaler


# ==================================================================================================================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folder')
parser.add_argument('--dataset')
args = parser.parse_args()
print('Arguments:', args)
folder_name = args.folder                 # 'Facebook100'
data_name = args.dataset                  # 'Bowdoin47', Bingham82
# python process_adj_emb.py --folder=Facebook100 --dataset=Bowdoin47

predicting_attribute = 'student_fac'      # 'gender', 'student_fac'
type = 'only_adj'                         # 'features_except', 'only_adj', 'embedding_of_adj'
# ==================================================================================================================================================================

# read adj and features from pickle and prepare sg graph
with open('../graph-data/{}/Processed/{}_featuresDf_hop_{}.pickle'.format(folder_name, data_name, str(0)), 'rb') as handle: features = pickle.load(handle) 
with open('../graph-data/{}/Processed/{}_adjDf_hop_{}.pickle'.format(folder_name, data_name, str(0)), 'rb') as handle: adj = pickle.load(handle)
with open('pickles/generated_embeddings/{}_{}.pickle'.format('Node2Vec', data_name), 'rb') as handle: emb = pickle.load(handle)
features_orig_nodes = list(features.index.values.tolist())

features = features[(features['student_fac'] > 0) & (features['major_index'] > 0) & (features['second_major'] > 0) & (features['dorm'] > 0) & (features['year'] > 0) & (features['gender'] > 0)]
print(features)
to_be_filtered = list(features.index.values.tolist())

for i in range(len(features_orig_nodes)):
    if str(i) not in to_be_filtered:
        # adj = adj.drop(str(i), axis=1) # col drop
        adj = adj.drop(str(i))         # row drop
        emb = emb.drop(str(i))         # row drop
        print(i)
adj = adj.reset_index(drop=True)
emb = emb.reset_index(drop=True)
print(emb)

with open('Filtered_nodes_edges/{}_adjDf_fullGraph_filterRow.pickle'.format(data_name), 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Filtered_nodes_edges/{}_embDf_fullGraph_filterRow.pickle'.format(data_name), 'wb') as handle: pickle.dump(emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('stored ... ')