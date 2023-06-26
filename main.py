# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from config import do_category_specific_task_prediction

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--analysis')
args = parser.parse_args()
print('Arguments:', args)
data_name = args.dataset   # 'Amherst41', 'American75', 'Bingham82', 'Bowdoin47'
analysis = args.analysis   # 'no_filtering', 'filtered_row', 'filtered_row_col'
print(data_name, analysis)
# python main.py --dataset=Bingham82 --analysis=no_filtering
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
selected_feature = ['fea', 'adj', 'emb']
# selected_feature = ['fea']
embedding = 'Node2Vec'

see = list()
for j in range(len(selected_feature)):
    res = do_category_specific_task_prediction(data_name, 'classification', 'RandomForest_hyper', 'student_fac', selected_feature[j], 15, embedding, analysis)
    # fes = do_category_specific_task_prediction(data_name, 'classification', 'RandomForest_hyper', 'gender', selected_feature[j], 15, embedding)
    see.append(selected_feature[j] + 'accuracy=' + str(res[0]) + 'f1_macro=' + str(res[1]))
    # print(res[1], 'done')
    # e_dict = dict()
    # matric_name = ['acc', 'f1_macro', 'precision_macro', 'recall_macro', 'f1_weighted', 'adj_RI']
    # for i in range(len(metric_name)):
    #     e_dict[i] = [matric_name[i], round(res[i], 6), round(fes[i], 6)]

    # resDf = pd.DataFrame.from_dict(e_dict, orient='index')
    # f = 'Result/' + data_name + '/' + data_name + '_category_' + str(j+1) + '_' + selected_feature[j] + '.xlsx'
    # resDf.to_excel(f) 
    # print(resDf)
for each in see:    
    print(each)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------