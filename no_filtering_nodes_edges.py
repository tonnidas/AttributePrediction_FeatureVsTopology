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
# python no_filtering_nodes_edges.py --folder=Facebook100 --dataset=Bingham82
embedding = 'Node2Vec'
predicting_attribute = 'student_fac'      # 'gender', 'student_fac'
type = 'only_adj'                         # 'features_except', 'only_adj', 'embedding_of_adj'
# ==================================================================================================================================================================

# read adj and features from pickle and prepare sg graph
with open('../graph-data/{}/Processed/{}_featuresDf_hop_{}.pickle'.format(folder_name, data_name, str(0)), 'rb') as handle: features = pickle.load(handle) 
with open('../graph-data/{}/Processed/{}_adjDf_hop_{}.pickle'.format(folder_name, data_name, str(0)), 'rb') as handle: adj = pickle.load(handle)
with open('pickles/generated_embeddings/{}_{}.pickle'.format(embedding, data_name), 'rb') as handle: emb = pickle.load(handle)

# All attributes = 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school  ('Status' = 'student_fac')
features = features.drop('high_school', axis=1)  # too many categories in high_school
features_orig_nodes = list(features.index.values.tolist())

# one hot encode all features
features_list = ['student_fac', 'gender', 'major_index', 'second_major', 'dorm']
for each_feature in features_list:
    if each_feature != predicting_attribute:
        oneHot = pd.get_dummies(features[each_feature], prefix = each_feature) # one hot encode the attribute
        features = features.join(oneHot)
        features = features.drop(each_feature, axis = 1)
features = features.reset_index(drop=True)


predicting_attr = features[predicting_attribute]
features = features.drop(predicting_attribute, axis=1)
print('Done features = ', features.shape, predicting_attr.shape, adj.shape, emb.shape)

with open('No_Filtered_nodes_edges/{}_hot_featuresDf_except_{}.pickle'.format(data_name, predicting_attribute), 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('No_Filtered_nodes_edges/{}_adjDf.pickle'.format(data_name), 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('No_Filtered_nodes_edges/{}_col_{}.pickle'.format(data_name, predicting_attribute), 'wb') as handle: pickle.dump(predicting_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)

