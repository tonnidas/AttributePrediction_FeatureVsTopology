# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import pickle
import os
from sklearn import metrics
from model import predict_attribute
# ----------------------------------------


# ----------------------------------------
dataset = 'Amherst41'                           # 'playgraph' or 'UNC28' or 'American75' or 'Amherst41' or 'Auburn71' or 'Baylor93' or 'Berkeley13' or 'Bingham82' or 'Bowdoin47' or 'Brandeis99' or 'Brown11' or 'BU10'
prediction_type = 'classification'                  # 'classification' or 'regression' 
model = 'RandomForest_hyper'                              # 'SVM' or 'RandomForest' or 'RandomForest_hyper'
predicting_attribute = 'gender'                     # 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school  ('Status' = 'student_fac')
selected_features = 'fea'                             # 'fea', 'adj', 'emb'
# Categories and their meaning in selected_features:
    # 'fea'          = Features
    # 'adj'          = Adjacency
    # 'emb'          = Embedding

rand_state_for_split = 15
embedding = 'Node2Vec'
# ----------------------------------------


# ----------------------------------------
# ----------------------------------------
def classification_metrics(Test_True_Labels, Test_Predicted_labels):
    acc = metrics.accuracy_score(Test_True_Labels, Test_Predicted_labels)
    f1_macro = metrics.f1_score(Test_True_Labels, Test_Predicted_labels, average='macro')
    precision_macro = metrics.precision_score(Test_True_Labels, Test_Predicted_labels, average='macro')
    recall_macro = metrics.recall_score(Test_True_Labels, Test_Predicted_labels, average='macro')
    f1_weighted = metrics.f1_score(Test_True_Labels, Test_Predicted_labels, average='weighted')
    adj_RI = metrics.adjusted_rand_score(Test_True_Labels, Test_Predicted_labels)
    
    return acc, f1_macro, precision_macro, recall_macro, f1_weighted, adj_RI
# ----------------------------------------
# ----------------------------------------


# ----------------------------------------
def constructor(dataset, predicting_attribute, embedding, selected_features, analysis):
    if analysis == 'no_filtering':
        with open('No_Filtered_nodes_edges/{}_hot_featuresDf_except_{}.pickle'.format(dataset, predicting_attribute), 'rb') as handle: featuresDf = pickle.load(handle) 
        with open('No_Filtered_nodes_edges/{}_adjDf.pickle'.format(dataset), 'rb') as handle: adjDf = pickle.load(handle) 
        with open('No_Filtered_nodes_edges/{}_col_{}.pickle'.format(dataset, predicting_attribute), 'rb') as handle: toPredictDf = pickle.load(handle)
        with open('pickles/generated_embeddings/{}_{}.pickle'.format(embedding, dataset), 'rb') as handle: embDf = pickle.load(handle)
        print('Done features = ', featuresDf.shape, toPredictDf.shape, adjDf.shape, embDf.shape)
    elif analysis == 'filtered_row':
        with open('Filtered_nodes_edges/{}_hot_featuresDf_except_{}.pickle'.format(dataset, predicting_attribute), 'rb') as handle: featuresDf = pickle.load(handle) 
        with open('Filtered_nodes_edges/{}_adjDf.pickle'.format(dataset), 'rb') as handle: adjDf = pickle.load(handle) 
        with open('Filtered_nodes_edges/{}_col_{}.pickle'.format(dataset, predicting_attribute), 'rb') as handle: toPredictDf = pickle.load(handle)
        with open('pickles/generated_embeddings/{}_{}_ones_3_only_adj.pickle'.format(dataset, embedding), 'rb') as handle: embDf = pickle.load(handle)
    elif analysis == 'filtered_row_col':  
        with open('Filtered_nodes_edges/{}_hot_featuresDf_except_{}.pickle'.format(dataset, predicting_attribute), 'rb') as handle: featuresDf = pickle.load(handle) 
        with open('Filtered_nodes_edges/{}_adjDf_fullGraph_filterRow.pickle'.format(dataset), 'rb') as handle: adjDf = pickle.load(handle) 
        with open('Filtered_nodes_edges/{}_col_{}.pickle'.format(dataset, predicting_attribute), 'rb') as handle: toPredictDf = pickle.load(handle)
        with open('Filtered_nodes_edges/{}_embDf_fullGraph_filterRow.pickle'.format(dataset), 'rb') as handle: embDf = pickle.load(handle)
    else: 
        print('wrong parameter')
        exit(0)

    if selected_features == 'fea': 
        return featuresDf, toPredictDf
    elif selected_features == 'adj':
        return adjDf, toPredictDf
    elif selected_features == 'emb':
        return embDf, toPredictDf
    else: 
        print('wrong prediction_type!')
        exit(0)
# ----------------------------------------


# ----------------------------------------
# Params:
#   dataset                 = a string containing dataset name
#   prediction_type         = a string ('classification' or 'regression')
#   model                   = a string
#   predicting_attribute    = a string
#   selected_features       = a string
#   rand_state_for_split    = an integer
# Return values:
#   acc, f1_macro, precision_macro, recall_macro, f1_weighted, adj_RI = a float
# def get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features):
def do_category_specific_task_prediction(dataset, prediction_type, model, predicting_attribute, selected_features, rand_state_for_split, embedding, analysis):
    
    featuresDf, y = constructor(dataset, predicting_attribute, embedding, selected_features, analysis)    # Get features and labels
    print("Featurs and y collected ___________________________", featuresDf.shape, y.shape)
    # result = predict_attribute(featuresDf, y, model, prediction_type, rand_state_for_split)

    # Get labels_test and labels_predicted
    y_test, predicted_labels = predict_attribute(featuresDf, y, model, prediction_type, rand_state_for_split)

    # Get evaluation metric values
    if prediction_type == 'classification':
        acc, f1_macro, precision_macro, recall_macro, f1_weighted, adj_RI = classification_metrics(y_test.tolist(), predicted_labels)
        print("acc: ", acc, "f1_macro: ", f1_macro, "precision_macro: ", precision_macro, "recall_macro: ", recall_macro, "f1_micro: ", "f1_weighted: ", f1_weighted, "adj_RI: ", adj_RI)

    return acc, f1_macro, precision_macro, recall_macro, f1_weighted, adj_RI
# ----------------------------------------