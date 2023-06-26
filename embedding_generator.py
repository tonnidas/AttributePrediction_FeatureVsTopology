# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import scipy.sparse 
from scipy.sparse import csr_matrix
import pickle
import stellargraph as sg
import os
from stellargraph import StellarGraph
from math import isclose
from sklearn.decomposition import PCA
from embedding_4_models import run



# ==================================================================================================================================================================
# Make the graph from the features and adj
def get_sg_graph(adj, features):
    print('adj shape:', adj.shape, 'feature shape:', features.shape)
    nxGraph = nx.from_scipy_sparse_array(adj)                           # make nx graph from scipy matrix

    # add features to nx graph
    for node_id, node_data in nxGraph.nodes(data=True):
        node_feature = features[node_id].todense()
        node_data["feature"] = np.squeeze(np.asarray(node_feature)) # convert to 1D matrix to array

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="gene", edge_type_default="connects to", node_features="feature")
    print(sgGraph.info())

    return sgGraph
# ==================================================================================================================================================================

# ==================================================================================================================================================================
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--folder')
parser.add_argument('--dataset')
parser.add_argument('--embedding')

args = parser.parse_args()
print('Arguments:', args)

folder_name = args.folder  # 'Facebook100'
data_name = args.dataset   # 'American75', 'Bowdoin47'
embedding = args.embedding # 'Node2Vec', 'GCN', 'Attri2Vec', 'GraphSAGE'
# python embedding_generator.py --folder=Facebook100 --dataset=Bowdoin47 --embedding=Node2Vec
predicting_attribute = 'student_fac'     # 'gender', 'student_fac'
# type = 'only_adj'                  # 'features_except_gender', 'properties_without_features', 'only_adj'
num_ones = 3
# ==================================================================================================================================================================



# read adj from pickle and prepare sg graph
with open('Filtered_nodes_edges/{}_adjDf.pickle'.format(data_name), 'rb') as handle: adjDf = pickle.load(handle) 
with open('Filtered_nodes_edges/{}_hot_featuresDf_except_{}.pickle'.format(data_name, predicting_attribute), 'rb') as handle: featuresDf = pickle.load(handle) 


if embedding == 'Node2Vec':
    adj = csr_matrix(adjDf.to_numpy())
    s = np.ones((adj.shape[0], num_ones))
    features = csr_matrix(s)

    # make StellarGraph and list of nodes
    sgGraph = get_sg_graph(adj, features)        # make the graph
    nodes_list = list(range(0, features.shape[0]))

    outputDf = run(embedding, data_name, nodes_list, sgGraph, 42, num_ones)
    outputFileName = "Embedding_scores/{}_{}_adj_roc_auc_featureOnes={}.txt".format(embedding, data_name, num_ones)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, 42, 0))
    f1.write(outputDf.to_string())
    f1.close()

else:  # embedding = 'GCN', 'Attri2Vec', 'GraphSAGE'
    print('building....')
    exit(0)
