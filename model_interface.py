import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fn
from scripts.models import Multi_Model5,Multi_Model5_plus
from scripts.datautils import *
from scripts.utils import *
from rdkit import Chem
import subprocess

#解决中文乱码问题
plt.rcParams['font.sans-serif'] = 'simhei'
plt.rcParams['axes.unicode_minus']=False
# 字体
TNR = {'fontname':'Arial'}

def get_parser(dataset):
    parser = argparse.ArgumentParser()
    # 数据基础参数

    parser.add_argument('--task_tag', required=False, default='scripts')
    parser.add_argument('--device', required=False, default=torch.device('cuda:0'))
    parser.add_argument('--data_shuffle', required=False, default=True)
    parser.add_argument('--predict_tag', required=False, default="dr_di", choices = ["dr_di","di_pr","dr_pr"],help='prediction')
    parser.add_argument('--model', required=False, default="model5+", choices=["model5", "model5+"],help='prediction')

    # 超参
    parser.add_argument('--k_fold', type=int, default=1, help='k-fold cross validation')
    parser.add_argument('--batch_size', type=int, default=5, help='number of batch_size')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=6, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')

    # 模型参数
    # deepwalk
    parser.add_argument('--embed_size', default=256, type=int, help='Embedding size')
    parser.add_argument('--num_hiddens', default=256, type=int, help='num hiddens')
    parser.add_argument('--out_dims', default=256, type=int, help='out_dims')
    parser.add_argument('--num_layers', default=2, type=int, help='num_layers')
    parser.add_argument('--num_heads', default=4, type=int, help='num_heads')
    parser.add_argument('--inputs_feats', default=0.4, type=float, help='Embedding size')

    # transformer
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='256', type=int, help='graph transformer output dimension')


    args = parser.parse_args()
    args.dataset=dataset
    args.data_dir = './data/' + args.dataset + '/'
    args.model_path = './checkpoint/' + args.dataset + '/best.pth'


    return args

def run_cmd(database):
    if database=="DrugBank":
        subprocess.call('makeblastdb -in ./data/DrugBank/DrugBank_add.fasta -dbtype prot',shell=True)
        subprocess.call('blastp -db ./data/DrugBank/DrugBank_add.fasta -query ./data/DrugBank/DrugBank_add.fasta -out ./data/DrugBank/protein_add_sim.tsv -outfmt 6', shell=True)


    elif database=="DAVIS":
        subprocess.call('makeblastdb -in ./data/DAVIS/DAVIS_add.fasta -dbtype prot  ',shell=True)
        subprocess.call('blastp -db ./data/DAVIS/DAVIS_add.fasta -query ./data/DAVIS/DAVIS_add.fasta -out ./data/DAVIS/protein_add_sim.tsv -outfmt 6',
                        shell=True)

    elif database=="KIBA":
        subprocess.call('makeblastdb -in ./data/KIBA/KIBA_add.fasta -dbtype prot  ',shell=True)
        subprocess.call(
            'blastp -db ./data/KIBA/KIBA_add.fasta -query ./data/KIBA/KIBA_add.fasta -out ./data/KIBA/protein_add_sim.tsv -outfmt 6',
            shell=True)




def load_dtidata(args,smi):
    data = {}
    drugs=pd.read_csv(args.data_dir+'Drug_infomation2.csv',header=0)
    proteins=pd.read_csv(args.data_dir+'Protein_infomation.csv',header=0)
    d_smiles=drugs['smile'].tolist()
    smi=Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    if args.dataset=='BrugBank':

        pro_id=proteins['protein'].tolist()
    else:
        pro_id=['protein'+str(i) for i in range(len(proteins))]
    data['protein_id']=pro_id
    data['sequence']=proteins['sequence'].tolist()
    if smi in d_smiles:
        did=d_smiles.index(smi)
        # sim_drug = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
        sim_drug=pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
        # sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()
        sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast.csv').iloc[:, 1:].to_numpy()
        data['sim_drug'] = sim_drug
        data['sim_protein'] = sim_protein
        drdr_graph, prpr_graph, data = dgl_similarity_graph(data, args)
        data['drug_sim_graphs'] = drdr_graph
        data['pro_sim_graphs'] = prpr_graph

        # 获取药物、疾病、蛋白特征信息
        output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv', 'graph2vec.csv', 'chemberta.csv']
        fdrugs_list = []
        for i in range(len(output_files)):
            if i == 3:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=None).to_numpy()

            elif i == 4 :
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=0).to_numpy()[:, 1:]
            else:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], sep=' ', header=None).to_numpy()
            fdrugs_list.append(fdrugs)

        data['drugfeature'] = fdrugs_list[4]
        data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM2.csv').to_numpy()[:,1:]



        args.drfeat_num = data['drugfeature'].shape[1]
        args.prfeat_num = data['proteinfeature'].shape[1]
        args.drug_number = data['drugfeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]

        data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)

        DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_train.txt', sep='\t', header=None)
        DW_embedding = DW_embedding.sort_values(0, ascending=True)
        DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)

        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)

        rela_dti=[]
        for i in range(args.protein_number):
            rela_dti.append([did,i])
        data['dr_pr']=np.array(rela_dti)
        return data,args

    else:

        Davis_drugsim(args.dataset,smi)  #添加新的药物并进行相似性矩阵生成

        sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast.csv').iloc[:, 1:].to_numpy()
        sim_drug = pd.read_csv(args.data_dir + 'DrugFingerprint_add.csv').iloc[:, 1:].to_numpy()
        # sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()

        data['sim_drug'] = sim_drug
        data['sim_protein'] = sim_protein
        drdr_graph, prpr_graph, data = dgl_similarity_graph_d(data, args)
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

        data['drugfeature'] = fdrugs_list[4]
        present_smi = genOneChemBERTA(smi)  # 获取输入药物的ChemBERTA特征
        data['drugfeature'] = np.concatenate([data['drugfeature'], present_smi], axis=0)

        data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM2.csv').to_numpy()[:,1:]

        args.drfeat_num = data['drugfeature'].shape[1]
        args.prfeat_num = data['proteinfeature'].shape[1]
        args.drug_number = data['drugfeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]



        data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)

        df = pd.read_csv(f'./data/{args.dataset}/DrugProteinCorrelation.csv', header=0)

        dpr = df.loc[df['label'] == 1]

        dr_pr = dpr.loc[:, ['drug', 'protein']].to_numpy()


        dr_pr[:, 1] = dr_pr[:, 1] + args.drug_number
        dr_dr = data['dr_dr']
        pr_pr = data['pr_pr'] + args.drug_number

        train_relation = np.concatenate([dr_pr, dr_dr, pr_pr], axis=0)
        df = pd.DataFrame(train_relation)
        df = df.sample(frac=1.0)
        df.to_csv(f'./data/{args.dataset}/Allrelation_full_add.txt', sep='\t', header=None, index=False)

        DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_full_add.txt', sep='\t', header=None)    #需要获取新的deepwalk
        DW_embedding = DW_embedding.sort_values(0, ascending=True)
        DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)

        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)
        did=args.drug_number-1
        rela_dti = []
        for i in range(args.protein_number):
            rela_dti.append([did, i])
        data['dr_pr'] = np.array(rela_dti)
        args.drug_number=args.drug_number-1
        return data, args





def load_tdidata(args,proseq):
    data = {}
    drugs=pd.read_csv(args.data_dir+'Drug_infomation2.csv',header=0)
    proteins=pd.read_csv(args.data_dir+'Protein_infomation.csv',header=0)
    pro_sequence=proteins['sequence'].tolist()
    smiles=drugs['smile'].values

    data['smiles']=smiles
    if args.dataset == 'BrugBank':
        data['drugs_id']=drugs['drug']
    else:
        data['drugs_id']=np.array(['drug_'+str(i) for i in range(len(drugs))])

    if proseq in pro_sequence:
        pid=pro_sequence.index(proseq)
        sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
        # sim_drug = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()

        sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast.csv').iloc[:, 1:].to_numpy()
        data['sim_drug'] = sim_drug
        data['sim_protein'] = sim_protein
        drdr_graph, prpr_graph, data = dgl_similarity_graph1(data, args)
        data['drug_sim_graphs'] = drdr_graph
        data['pro_sim_graphs'] = prpr_graph

        # 获取药物、疾病、蛋白特征信息
        output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv', 'graph2vec.csv', 'chemberta.csv']
        fdrugs_list = []
        for i in range(len(output_files)):
            if i == 3:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=None).to_numpy()

            elif i == 4 or i == 5:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=0).to_numpy()[:, 1:]
            else:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], sep=' ', header=None).to_numpy()
            fdrugs_list.append(fdrugs)

        data['drugfeature'] = fdrugs_list[4]
        data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM2.csv').to_numpy()[:,1:]



        args.drfeat_num = data['drugfeature'].shape[1]
        args.prfeat_num = data['proteinfeature'].shape[1]
        args.drug_number = data['drugfeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]

        data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)

        DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_train.txt', sep='\t', header=None)
        DW_embedding = DW_embedding.sort_values(0, ascending=True)
        DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)

        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)

        rela_tdi=[]
        for i in range(args.drug_number):
            rela_tdi.append([i,pid])
        data['dr_pr']=np.array(rela_tdi)
        return data,args

    else:
        #获得蛋白序列的fasta文件
        get_profasta(args.dataset,proseq)

        #使用blast计算蛋白相似性,调用subprocess

        run_cmd(args.dataset)

        #读取文件计算相似性矩阵
        get_prosim_matrix(args.dataset)

        sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast_add.csv').iloc[:, 1:].to_numpy()
        sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
        # sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()

        data['sim_drug'] = sim_drug
        data['sim_protein'] = sim_protein
        drdr_graph, prpr_graph, data = dgl_similarity_graph_p(data, args)
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

            elif i == 4 or i==5:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=0).to_numpy()[:, 1:]
            else:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], sep=' ', header=None).to_numpy()
            fdrugs_list.append(fdrugs)

        data['drugfeature'] = fdrugs_list[4]


        data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM2.csv').to_numpy()[:,1:]
        pro_esm=genOneProteinESM2(proseq)
        data['proteinfeature']=np.concatenate([data['proteinfeature'],pro_esm],axis=0)

        args.drfeat_num = data['drugfeature'].shape[1]
        args.prfeat_num = data['proteinfeature'].shape[1]
        args.drug_number = data['drugfeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]



        data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)

        # df = pd.read_csv(f'./data/{args.dataset}/DrugProteinCorrelation.csv', header=0)
        #
        # dpr = df.loc[df['label'] == 1]
        #
        # dr_pr = dpr.loc[:, ['drug', 'protein']].to_numpy()
        #
        #
        # dr_pr[:, 1] = dr_pr[:, 1] + args.drug_number
        # dr_dr = data['dr_dr']
        # pr_pr = data['pr_pr'] + args.drug_number
        #
        # train_relation = np.concatenate([dr_pr, dr_dr, pr_pr], axis=0)
        # df = pd.DataFrame(train_relation)
        # df = df.sample(frac=1.0)
        # df.to_csv(f'./data/{args.dataset}/Allrelation_full_padd.txt', sep='\t', header=None, index=False)


        DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_full_padd.txt', sep='\t', header=None)
        DW_embedding = DW_embedding.sort_values(0, ascending=True)
        DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)

        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)
        pid=args.protein_number-1
        rela_tdi = []
        for i in range(args.drug_number):
            rela_tdi.append([i, pid])
        data['dr_pr'] = np.array(rela_tdi)
        args.neighbor=6
        args.protein_number=args.protein_number-1
        return data, args


def load_Dis2Drugdata(args,dis):
    data = {}

    rawData=pd.read_csv(args.data_dir+'node_data.csv')
    drugs=rawData[rawData['type']=='drug']
    diseases=rawData[rawData['type'] == 'disease']
    dis_names=diseases['name'].tolist()
    smiles=drugs['smiles'].values


    data['drugs_id'] = drugs['name'].values
    data['smiles']=smiles

    if dis in dis_names:
        disid=dis_names.index(dis)
        sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
        # sim_drug = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
        sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()
        # sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast.csv').iloc[:, 1:].to_numpy()
        data['sim_drug'] = sim_drug
        data['sim_disease'] = sim_disease
        drdr_graph, didi_graph, data = dgl_similarity_graph2(data, args)
        data['drug_sim_graphs'] = drdr_graph
        data['dis_sim_graphs'] = didi_graph

        # 获取药物、疾病特征信息
        output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv', 'graph2vec.csv', 'chemberta.csv']
        fdrugs_list = []
        for i in range(len(output_files)):
            if i == 3:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=None).to_numpy()

            elif i==4 or i==5:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], header=0).to_numpy()[:, 1:]
            else:
                fdrugs = pd.read_csv(args.data_dir + "drugs/" + output_files[i], sep=' ', header=None).to_numpy()
            fdrugs_list.append(fdrugs)

        data['drugfeature'] = fdrugs_list[4]
        data['diseasefeature'] = pd.read_csv(args.data_dir + 'Disease_MESH.csv')
        data['diseasefeature'].drop(data['diseasefeature'].columns[1], axis=1, inplace=True)
        data['diseasefeature'] = data['diseasefeature'].to_numpy()

        args.drfeat_num = data['drugfeature'].shape[1]
        args.difeat_num = data['diseasefeature'].shape[1]
        args.drug_number = data['drugfeature'].shape[0]
        args.disease_number = data['diseasefeature'].shape[0]

        data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        data['di_feature'] = torch.FloatTensor(data['diseasefeature']).to(args.device)



        DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_train.txt', sep='\t', header=None)
        DW_embedding = DW_embedding.sort_values(0, ascending=True)
        DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)
        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)

        rela_ddi=[]
        for i in range(args.drug_number):
            rela_ddi.append([i,disid])
        data['dr_di']=np.array(rela_ddi)
        return data,args

    else:
        pass
        # index=all_dname.index(dis)
        # #dataset 中其它疾病的id号
        # data_dis=[int(i[1:]) for i in diseases['id']]
        # data_dis=[all_did.index(d) for d in data_dis]
        #
        # get_disease_sim(index,data_dis)
        #
        # #获得疾病的相似性
        #
        #
        #
        # # sim_protein = pd.read_csv(args.data_dir + 'SimProtein_blast_add.csv').iloc[:, 1:].to_numpy()
        # sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
        # sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()
        #
        # data['sim_drug'] = sim_drug
        # data['sim_disease'] = sim_disease
        # drdr_graph, prpr_graph, data = dgl_similarity_graph_p(data, args)
        # data['drug_sim_graphs'] = drdr_graph
        # data['pro_sim_graphs'] = prpr_graph
        #
        # # 进行相似性网络，关系获取
        # data['dr_dr'] = get_smi_relation(drdr_graph)
        # data['pr_pr'] = get_smi_relation(prpr_graph)
        #
        # # 获取药物、疾病特征信息
        # output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv', 'graph2vec.csv', 'chemberta.csv', 'unimol.csv']
        # fdrugs_list = []
        # for i in range(len(output_files)):
        #     if i == 3:
        #         fdrugs = pd.read_csv(args.data_dir + "drug/" + output_files[i], header=None).to_numpy()
        #
        #     elif i == 4 or i == 5:
        #         fdrugs = pd.read_csv(args.data_dir + "drug/" + output_files[i], header=0).to_numpy()[:, 1:]
        #     else:
        #         fdrugs = pd.read_csv(args.data_dir + "drug/" + output_files[i], sep=' ', header=None).to_numpy()
        #     fdrugs_list.append(fdrugs)
        #
        # data['drugfeature'] = fdrugs_list[4]
        #
        # data['diseasefeature'] = pd.read_csv(args.data_dir + 'Disease_MESH.csv')
        # data['diseasefeature'].drop(data['diseasefeature'].columns[1], axis=1, inplace=True)
        # data['diseasefeature'] = data['diseasefeature'].to_numpy()
        # dis_mesh=0
        # data['proteinfeature']=np.concatenate([data['proteinfeature'],dis_mesh],axis=0)
        #
        # args.drfeat_num = data['drugfeature'].shape[1]
        # args.bfeat_num = data['proteinfeature'].shape[1]
        # args.drug_number = data['drugfeature'].shape[0]
        # args.b_number = data['proteinfeature'].shape[0]
        #
        #
        #
        # data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        # data['pr_feature'] = torch.FloatTensor(data['proteinfeature']).to(args.device)
        #
        # df = pd.read_csv(f'./data/{args.dataset}/DrugProteinCorrelation.csv', header=0)
        #
        # dpr = df.loc[df['label'] == 1]
        #
        # dr_pr = dpr.loc[:, ['drug', 'protein']].to_numpy()
        #
        #
        # dr_pr[:, 1] = dr_pr[:, 1] + args.drug_number
        # dr_dr = data['dr_dr']
        # pr_pr = data['pr_pr'] + args.drug_number
        #
        # train_relation = np.concatenate([dr_pr, dr_dr, pr_pr], axis=0)
        # df = pd.DataFrame(train_relation)
        # df = df.sample(frac=1.0)
        # df.to_csv(f'./data/{args.dataset}/Allrelation_full_padd.txt', sep='\t', header=None, index=False)
        # subprocess.call('conda activate pytorch_38',shell=True)
        #
        #
        # DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_full_padd.txt', sep='\t', header=None)
        # DW_embedding = DW_embedding.sort_values(0, ascending=True)
        # DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)
        #
        # data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)
        # pid=args.b_number-1
        # rela_tdi = []
        # for i in range(args.drug_number):
        #     rela_tdi.append([i, pid])
        # data['dr_pr'] = np.array(rela_tdi)
        # args.neighour=6
        # return data, args

def load_ddidata(args,smi):
    data = {}

    df=pd.read_csv(args.data_dir+'node_data.csv',header=0)

    disease=df[df['type'] == 'disease']

    drugs = df[df['type'] == 'drug']
    d_smiles = drugs['smiles'].tolist()
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    data['disease_id']=disease['name'].values
    if smi in d_smiles:
        did=d_smiles.index(smi)
        sim_drug = pd.read_csv(args.data_dir + 'SimDrug.csv').iloc[:, 1:].to_numpy()
        sim_disease = pd.read_csv(args.data_dir + 'SimDisease.csv').iloc[:, 1:].to_numpy()
        data['sim_drug'] = sim_drug
        data['sim_protein'] = sim_disease
        drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
        data['drug_sim_graphs'] = drdr_graph
        data['dis_sim_graphs'] = didi_graph

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

        data['drugfeature'] = fdrugs_list[4]
        data['diseasefeature'] = pd.read_csv(args.data_dir + 'Disease_MESH.csv')
        data['diseasefeature'].drop(data['diseasefeature'].columns[1], axis=1, inplace=True)
        data['diseasefeature'] = data['diseasefeature'].to_numpy()



        args.drfeat_num = data['drugfeature'].shape[1]
        args.bfeat_num = data['diseasefeature'].shape[1]
        args.drug_number = data['drugfeature'].shape[0]
        args.b_number = data['diseasefeature'].shape[0]

        data['dr_feature'] = torch.FloatTensor(data['drugfeature']).to(args.device)
        data['di_feature'] = torch.FloatTensor(data['diseasefeature']).to(args.device)



        # data['DW_embedding'] = 0
        DW_embedding = pd.read_csv(args.data_dir + 'Allrelation_train_00.txt.txt', sep='\t', header=None)
        DW_embedding = DW_embedding.sort_values(0, ascending=True)
        DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)

        rela_ddi=[]
        for i in range(args.b_number):
            rela_ddi.append([did,i])
        data['dr_di']=np.array(rela_ddi)
        return args,data

    else:
        pass


def load_model_ddi(args,dataset):
    if dataset == 'C-dataset':
        model = Multi_Model5_plus(args, drug_in_dim=args.drfeat_num,
                                  target_in_dim=args.difeat_num,
                                  embed_size=args.embed_size,
                                  num_hiddens=args.num_hiddens,
                                  num_layers=args.num_layers,
                                  num_heads=args.num_heads,
                                  labels=2,
                                  gt_layer=args.gt_layer,
                                  gt_head=args.gt_head,
                                  gt_out_dim=args.gt_out_dim,
                                  dropout=args.dropout,
                                  num_drugs=args.drug_number,
                                  num_targets=args.disease_number)
    else:
        model = Multi_Model5(args, drug_in_dim=args.drfeat_num,
                                 target_in_dim=args.difeat_num,
                                 embed_size=args.embed_size,
                                 num_hiddens=args.num_hiddens,
                                 num_layers=args.num_layers,
                                 num_heads=args.num_heads,
                                 labels=2,
                                 gt_layer=args.gt_layer,
                                 gt_head=args.gt_head,
                                 gt_out_dim=args.gt_out_dim,
                                 dropout=args.dropout,
                                 num_drugs=args.drug_number,
                                 num_targets=args.disease_number)
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)

    return model

def load_model_dti(args,dataset):

    if dataset=='DrugBank':
        model = Multi_Model5_plus(args, drug_in_dim=args.drfeat_num,
                             target_in_dim=args.prfeat_num,
                             embed_size=args.embed_size,
                             num_hiddens=args.num_hiddens,
                             num_layers=args.num_layers,
                             num_heads=args.num_heads,
                             labels=2,
                             gt_layer=args.gt_layer,
                             gt_head=args.gt_head,
                             gt_out_dim=args.gt_out_dim,
                             dropout=args.dropout,
                             num_drugs=args.drug_number,
                             num_targets=args.protein_number)

    else:
        model = Multi_Model5(args, drug_in_dim=args.drfeat_num,
                             target_in_dim=args.prfeat_num,
                             embed_size=args.embed_size,
                             num_hiddens=args.num_hiddens,
                             num_layers=args.num_layers,
                             num_heads=args.num_heads,
                             labels=2,
                             gt_layer=args.gt_layer,
                             gt_head=args.gt_head,
                             gt_out_dim=args.gt_out_dim,
                             dropout=args.dropout,
                             num_drugs=args.drug_number,
                             num_targets=args.protein_number)

    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)

    return model




def draw_picture(dataset,x_name,y):

    # 设置画布大小
    plt.rc('font', family='Arial')
    plt.figure(figsize=(10,6))
    figure, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=1000)

    colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
              "#86BCB6", "#E15759", "#E19D9A"]
    x=range(len(x_name))
    plt.barh(x, y,color=colors)
    for i in x:
        plt.text(y[i]+0.05, x[i], s=str(round(y[i]*100,2))+"%",va='center',fontsize=12,**TNR)
    plt.title(f"Strong correlation results for {dataset}(TOP10)", fontsize=20)
    # 为了美观，不显示画布的黑色边框
    [axes.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]
    # 不显示x轴坐标
    axes.set_xticks([])

    plt.yticks(x, x_name,fontsize=12)
    return plt


def DTI(smi,check,max_num):
    args = get_parser(check)

    data,args=load_dtidata(args,smi)
    model=load_model_dti(args,check)
    model.eval()
    xtest=torch.LongTensor(data['dr_pr'])
    with torch.no_grad():
        test_preds = model(data['dr_feature'], data['pr_feature'], data['drug_sim_graphs'],
                           data['pro_sim_graphs'], data['DW_embedding'], xtest)
        test_prob = fn.softmax(test_preds, dim=-1)[:, 1].cpu().numpy()
    res_id=test_prob.argsort()[::-1]
    pnames=[data['protein_id'][i] for i in res_id]
    pscores=list(test_prob[res_id])
    sequence=[data['sequence'][i] for i in res_id]
    pro_res = pd.DataFrame({'Protein ID':pnames, 'Score':pscores, 'FASTA':sequence})

    pro_res = pro_res[:max_num]
    plt=draw_picture(args.dataset,pnames[:12][::-1],pscores[:12][::-1])
    plt.savefig(r".\test1.png", format="png")
    return plt,pro_res

def TDI(pro,check,max_num):
    args = get_parser(check)

    data,args=load_tdidata(args,pro)
    model=load_model_dti(args,check)
    model.eval()
    xtest=torch.LongTensor(data['dr_pr'])
    with torch.no_grad():
        test_preds = model(data['dr_feature'], data['pr_feature'], data['drug_sim_graphs'],
                           data['pro_sim_graphs'], data['DW_embedding'], xtest)
        test_prob = fn.softmax(test_preds, dim=-1)[:, 1].cpu().numpy()
    res_id=list(test_prob.argsort()[::-1])
    dnames=data['drugs_id'][res_id]
    dscores=list(test_prob[res_id])
    dsmiles=data['smiles'][res_id]
    drug_res = pd.DataFrame({'Drug ID':dnames, 'Score':dscores, 'SMILES':dsmiles})

    drug_res.to_csv("./result/result_TDI_" + pro[:5] + ".csv")
    drug_res.sort_values(by='Score', inplace=True, ascending=False)
    drug_res = drug_res[:max_num]
    plt=draw_picture(args.dataset,dnames[:12][::-1],dscores[:12][::-1])
    plt.savefig(r".\test2.png", format="png")
    return plt,drug_res


def Dis2Drug(dis,check,max_num):
    args = get_parser(check)
    max_num = int(max_num)
    data,args=load_Dis2Drugdata(args,dis)
    model=load_model_ddi(args,check)
    model.eval()
    xtest=torch.LongTensor(data['dr_di'])
    with torch.no_grad():
        test_preds = model(data['dr_feature'], data['di_feature'], data['drug_sim_graphs'],
                           data['dis_sim_graphs'], data['DW_embedding'], xtest)
        test_prob = fn.softmax(test_preds, dim=-1)[:, 1].cpu().numpy()
    res_id=test_prob.argsort()[::-1]
    dnames = data['drugs_id'][res_id]
    dscores = list(test_prob[res_id])
    dsmiles = data['smiles'][res_id]
    drug_res = pd.DataFrame({'Drug ID':dnames, 'Score':dscores, 'SMILES':dsmiles})


    drug_res.to_csv('./result/predicted_drugs_scores.csv',index=False)
    drug_res.sort_values(by='Score', inplace=True, ascending=False)
    drug_res = drug_res[:max_num]

    plt=draw_picture(args.dataset,dnames[:12][::-1],dscores[:12][::-1])
    plt.savefig(r".\test3.png", format="png",dpi=800)
    return plt,drug_res

def DDI(smi, check,max_num):
    args = get_parser(check)
    print(args)
    data, args = load_ddidata(args, smi)
    model = load_model_ddi(args, check)
    xtest = torch.LongTensor(data['dr_di'])
    test_preds = model(data['dr_feature'], data['di_feature'], data['drug_sim_graphs'],
                       data['di_sim_graphs'], data['DW_embedding'], xtest)
    test_prob = fn.softmax(test_preds, dim=-1)[:, 1].cpu().numpy()
    res_id = test_prob.argsort()[::-1]

    dnames = data['drugs_id'][res_id]
    dscores = list(test_prob[res_id])
    dis_names=data['disease_id'][res_id]
    disease_res = pd.DataFrame({'Disease name': dis_names, 'Score': dscores})
    disease_res = disease_res[:max_num]

    plt = draw_picture(args.dataset, dis_names[:12][::-1], dscores[:12][::-1])
    plt.savefig(r".\test4.png", format="png")
    return plt,disease_res




if __name__ == '__main__':
    #30 davis
    #5 kiba COC1C(N(C)C(=O)c2cccc([N+](=O)[O-])c2)CC2OC1(C)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4
    '''
    F-dataset:model5_plus  *   best.pth
    B-dataset:model5   *   best.pth 
    C-dataset:model5_plus *
    
    DrugBank:model5_plus  *
    DAVIS:model5   +/-
    KIBA: model5  *
    '''


    # smi='Nc1nc(Nc2ccc(S(N)(=O)=O)cc2)nn1C(=S)Nc1c(F)cccc1F'
    smi2='CN1C2CC(CC1C3C2O3)OC(=O)C(CO)C4=CC=CC=C4'   #Scopolamine
    # pro1='EQLLPDLLISPHMLPLTDLEIKFQYRGRPPRALTISNPHGCRLFYSQLEATQEQVELFGPISLEQVRFPSPEDIPSDKQRFYTNQLLDVLDRGLILQLQGQDLYAIRLCQCKVFWSGPCASAHDSCPNPIQREVKTKLFSLEHFLNELILFQKGQTNTPPPFEIFFCFGEEWPDRKPREKKLITVQVVPVAARLLLEMFSGELSWSADDIRLQISNPDLKDRMVEQFKELHHIWQSQQRLQPVAQA'
    # pro='MDGETAEEQGGPVPPPVAPGGPGLGGAPGGRREPKKYAVTDDYQLSKQVLGLGVNGKVLECFHRRTGQKCALKLLYDSPKARQEVDHHWQASGGPHIVCILDVYENMHHGKRCLLIIMECMEGGELFSRIQERGDQAFTEREAAEIMRDIGTAIQFLHSHNIAHRDVKPENLLYTSKEKDAVLKLTDFGFAKETTQNALQTPCYTPYYVAPEVLGPEKYDKSCDMWSLGVIMYILLCGFPPFYSNTGQAISPGMKRRIRLGQYGFPNPEWSEVSEDAKQLIRLLLKTDPTERLTITQFMNHPWINQSMVVPQTPLHTARVLQEDKDHWDEVKEEMTSALATMRVDYDQVKIKDLKTSNNRLLNKRRKKQAGSSSASQGCNNQ'
    pro_res=DTI(smi2,'KIBA',10)


    # plt.show()