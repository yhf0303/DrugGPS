import os.path

import pandas as pd
import numpy as np
import random
import torch
import pickle
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

'''
dr = drug
di = disease
pr = protein
'''
def get_sparse(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size)
    adj.to_dense().long()
    return adj

def get_smi_relation(graph):
    dr_dr = graph.edges()

    dr_dr = torch.stack(dr_dr, dim=1).numpy()
    dr_dr_list = []
    for i in range(len(dr_dr)):
        if dr_dr[i][0] == dr_dr[i][1]:
            continue
        else:
            dr_dr_list.append(dr_dr[i])

    return np.array(dr_dr_list)


def load_DTIdata(args):
    data = dict()

    data = data_processing1(data, args)
    # # 相似性网络数据
    # sim_drug = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
    sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast.csv').iloc[:, 1:].to_numpy()
    data['sim_drug'] = sim_drug
    data['sim_protein'] = sim_protein
    drdr_graph, prpr_graph, data = dgl_similarity_graph1(data, args)
    data['drug_sim_graphs'] = drdr_graph
    data['pro_sim_graphs'] = prpr_graph

    # 进行相似性网络，关系获取
    data['dr_dr'] = get_smi_relation(drdr_graph)
    data['pr_pr'] = get_smi_relation(prpr_graph)

    # 获取药物、疾病、蛋白特征信息
    output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv', 'graph2vec.csv', 'chemberta.csv']
    fdrugs_list = []
    for i in range(len(output_files)):
        if i == 3:
            fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=None).to_numpy()

        elif i == 4:
            fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=0).to_numpy()[:, 1:]
        else:
            fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], sep=' ', header=None).to_numpy()
        fdrugs_list.append(fdrugs)
    # data['drugfeature'] = np.concatenate(fdrugs_list[:4], axis=1)
    data['drugfeature'] = fdrugs_list[4]
    data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM2.csv').to_numpy()[:, 1:]

    args.drfeat_num = data['drugfeature'].shape[1]
    # args.difeat_num = data['diseasefeature'].shape[1]
    args.prfeat_num = data['proteinfeature'].shape[1]
    args.drug_number = data['drugfeature'].shape[0]
    # args.disease_number = data['diseasefeature'].shape[0]
    args.protein_number = data['proteinfeature'].shape[0]

    data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
    # data['di_feature'] = torch.FloatTensor(data['diseasefeature']).to(args.device)
    data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)

    return data, args


def load_DDIdata(args):
    data = dict()
    # # 相似性网络数据
    sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
    sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()

    data['sim_drug'] = sim_drug
    data['sim_disease'] = sim_disease

    drdr_graph, didi_graph, data = dgl_similarity_graph2(data, args)
    data['drug_sim_graphs'] = drdr_graph
    data['dis_sim_graphs'] = didi_graph


    # 进行相似性网络，关系获取
    data['dr_dr'] = get_smi_relation(drdr_graph)

    data['di_di'] = get_smi_relation(didi_graph)

    # 关系网络数据
    drug_disease = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int, usecols=[0, 1]).to_numpy()
    disease_protein = pd.read_csv(args.data_dir + 'DiseaseProteinAssociationNumber.csv', dtype=int,
                                  usecols=[0, 1]).to_numpy()
    drug_protein = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int, usecols=[0, 1]).to_numpy()

    data['dr_di'] = drug_disease
    data['di_pr'] = disease_protein
    data['dr_pr'] = drug_protein



    # data['tp_factor'] = tp_factor

    # 获取药物、疾病、蛋白特征信息
    output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv','chemberta.csv', 'graph2vec.csv']
    fdrugs_list = []
    for i in range(len(output_files)):
        if i == 4:
            fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=None).to_numpy()
        elif i == 3:
            fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=0).to_numpy()[:, 1:]
        else:
            fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], sep=' ', header=None).to_numpy()
        fdrugs_list.append(fdrugs)
    # data['drugfeature'] = np.concatenate(fdrugs_list[:4], axis=1)
    data['drugfeature'] = fdrugs_list[3]

    data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM2.csv').to_numpy()[:, 1:]
    data['diseasefeature'] = pd.read_csv(args.data_dir + 'Disease_MESH.csv')
    data['diseasefeature'].drop(data['diseasefeature'].columns[1], axis=1, inplace=True)
    data['diseasefeature'] = data['diseasefeature'].to_numpy()

    args.drfeat_num = data['drugfeature'].shape[1]
    args.difeat_num = data['diseasefeature'].shape[1]
    args.prfeat_num = data['proteinfeature'].shape[1]
    args.drug_number = data['drugfeature'].shape[0]
    args.disease_number = data['diseasefeature'].shape[0]
    args.protein_number = data['proteinfeature'].shape[0]

    data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
    data['di_feature'] = torch.FloatTensor(data['diseasefeature']).to(args.device)
    # data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)



    data = data_processing2(data, args)

    # for i in range(args.k_fold):
    #     xtrain=data['X_train'][i]
    #     xtest=data['X_test'][i]
    #     ytrain=data['Y_train'][i]
    #     ytest=data['Y_test'][i]
    #     xytrain=np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    #     xytest=np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    #     np.save(f'./data/{args.dataset}/{i}_dr_di_train.npy',xytrain)
    #
    #     np.save(f'./data/{args.dataset}/{i}_dr_di_test.npy', xytest)
    #
    #     relation_train = []
    #     for k, lab in enumerate(ytrain):
    #         if lab == 1:
    #             relation_train.append(xtrain[k])
    #     dr_di = np.array(relation_train)
    #     dr_di[:,1] = dr_di[:,1]+args.drug_number
    #     di_pr=data['di_pr']
    #     di_pr[:,0]=di_pr[:,0]+args.drug_number
    #     di_pr[:,1]=di_pr[:,1]+args.drug_number+args.disease_number
    #     dr_pr=data['dr_pr']
    #     dr_pr[:, 1] = dr_pr[:, 1] + args.drug_number+args.disease_number
    #     dr_dr=data['dr_dr']
    #     di_di=data['di_di']+args.drug_number
    #
    #
    #
    #     train_relation=np.concatenate([dr_di,dr_pr,di_pr,dr_dr,di_di],axis=0)
    #     df=pd.DataFrame(train_relation)
    #     df = df.sample(frac=1.0)
    #     df.to_csv(f'./data/{args.dataset}/Allrelation_train.txt', sep='\t' ,header=None,index=False)


    return data, args

def data_processing1(data, args):
    # 处理Y值 平衡正负样本

    df=pd.read_csv(f'./data/{args.dataset}/DrugProteinCorrelation.csv',header=0)

    dr_pr=df.loc[df['label']==1]

    data['dr_pr']=dr_pr.loc[:,['drug','protein']].to_numpy()

    X=df.iloc[:,:2].values  #34767,2
    Y=df['label'].values

    X_train_all=[]
    X_test_all=[]
    Y_train_all=[]
    Y_test_all=[]


    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=12, test_size=0.2, stratify=Y)
    X_train_all.append(x_train)
    X_test_all.append(x_test)
    Y_train_all.append(y_train)
    Y_test_all.append(y_test)

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def data_processing2(data, args):
    # 处理Y值 平衡正负样本
    if args.predict_tag == "dr_di":
        matrix = get_sparse(data[args.predict_tag], (args.drug_number, args.disease_number))
    elif args.predict_tag == "di_pr":
        matrix = get_sparse(data[args.predict_tag], (args.disease_number, args.protein_number))
    else:
        matrix = get_sparse(data[args.predict_tag], (args.drug_number, args.protein_number))
    one_index = []
    zero_index = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    # 采样的数据
    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    X = np.array(one_index + zero_index, dtype=int)
    Y = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)

    k = args.k_fold
    # skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=args.data_shuffle)
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []
    # for train_index, test_index in skf.split(X, Y):
    #     # print('Train:', train_index, 'Test:', test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]
    #     Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
    #     Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
    #     X_train_all.append(X_train)
    #     X_test_all.append(X_test)
    #     Y_train_all.append(Y_train)
    #     Y_test_all.append(Y_test)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=12, test_size=0.2, stratify=Y)
    X_train_all.append(x_train)
    X_test_all.append(x_test)
    Y_train_all.append(y_train)
    Y_test_all.append(y_test)

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def get_metric(y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    Auc = auc(fpr, tpr)

    precision1, recall1, _ = precision_recall_curve(y_true, y_prob)
    Aupr = auc(recall1, precision1)

    return Auc, Aupr, accuracy, precision, recall, f1, mcc

def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def dgl_similarity_graph1(data, args):
    drdr_matrix = k_matrix(data['sim_drug'], args.neighbor)

    prpr_matrix = k_matrix(data['sim_protein'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)

    prpr_nx = nx.from_numpy_matrix(prpr_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)

    prpr_graph = dgl.from_networkx(prpr_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['sim_drug'])

    prpr_graph.ndata['drs'] = torch.tensor(data['sim_protein'])

    return drdr_graph, prpr_graph, data


def dgl_similarity_graph2(data, args):
    drdr_matrix = k_matrix(data['sim_drug'], args.neighbor)
    didi_matrix = k_matrix(data['sim_disease'], args.neighbor)

    drdr_nx = nx.from_numpy_matrix(drdr_matrix)
    didi_nx = nx.from_numpy_matrix(didi_matrix)

    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)


    drdr_graph.ndata['drs'] = torch.tensor(data['sim_drug'])
    didi_graph.ndata['drs'] = torch.tensor(data['sim_disease'])

    return drdr_graph, didi_graph,data