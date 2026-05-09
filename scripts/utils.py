import esm
import torch
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModelForMaskedLM
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
from rdkit import Chem
import os
import csv
import dgl
import networkx as nx

def genOneChemBERTA(smile):
    tokenizer = AutoTokenizer.from_pretrained("./CHEMBERTA_2")
    model = AutoModelForMaskedLM.from_pretrained("./CHEMBERTA_2")
    inputs = tokenizer(smile, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        results = model(**inputs)
    # batch_lens=torch.tensor(results.logits.shape[1])
    all_data = []
    for i in range(results.logits.shape[0]):
        token_representations = results.logits[i, :].mean(0).numpy()
        # token_representations.insert(0,smile_index[i])
        all_data.append(token_representations.reshape(1,-1))

    return all_data[0]



def get_morgan_fingerprint(mol):
    x=AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

    return x

def genOneProteinESM2(proseq):
    model,alphabet=esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter=alphabet.get_batch_converter()
    all_data=[]
    data=[('1',proseq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens=(batch_tokens!=alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results=model(batch_tokens, repr_layers=[12], return_contacts=True)
    token_representations=results["representations"][12][0, 1: batch_lens[0]-1].mean(0).numpy()
    return token_representations.reshape(1,-1)


def Davis_drugsim(dataset,smi):
    df1 = pd.read_csv(f'./data/{dataset}/Drug_infomation.csv')
    output_path = f'./data/{dataset}/DrugFingerprint_add.csv'
    df=pd.DataFrame()
    smi_list=np.append(df1['smile'],smi)
    df['smile']=smi_list
    df['mol'] = df['smile'].apply(lambda x: Chem.MolFromSmiles(x))
    print(df[df['mol'].isnull()])
    df = df[df['mol'].notnull()]
    x = AllChem.GetMorganFingerprintAsBitVect(df['mol'][0], 2, 2048)
    df['fingerprint'] = df['mol'].apply(get_morgan_fingerprint)
    fps = df['fingerprint'].tolist()
    # 计算相似性矩阵中的每个元素
    n = len(fps)
    similarity_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="Calculating similarities"):
        for j in range(i, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # 相似度矩阵是对称的

    similarity_matrix = pd.DataFrame(similarity_matrix)
    similarity_matrix.to_csv(output_path, index=True)


def get_profasta(database,inp_seq):

    if database == "B-dataset" or database == "C-dataset" or database == "F-dataset":
        filename = os.path.join("./data", database, database[:1] + "_data.csv")
        outname = database + ".fasta"
        outpath = os.path.join("./data", database, outname)
        csvfile = open(filename, 'r')
        reader = csv.reader(csvfile)
        next(reader)
        num = 0
        with open(outpath, 'w', encoding="UTF-8") as outfile:
            for i in reader:
                if i[2] == "protein":
                    num += 1
                    outfile.write(">{}\n".format(i[0]))
                    seq = i[5]
                    split_line = [seq[i:i + 70] for i in range(0, len(seq), 70)]
                    for j in split_line:
                        outfile.write("{}\n".format(j))
                    outfile.write("\n".format(j))

            outfile.write(">{}\n".format(i[0]))
            seq = inp_seq
            split_line = [seq[i:i + 70] for i in range(0, len(seq), 70)]
            for j in split_line:
                outfile.write("{}\n".format(j))
            outfile.write("\n".format(j))


    if database=='DrugBank':
        filename = os.path.join("./data", database, "Protein_infomation.csv")
        outname = database + "_add.fasta"
        outpath = os.path.join("./data", database, outname)
        csvfile = open(filename, 'r')
        reader = csv.reader(csvfile)
        next(reader)
        num = 0
        with open(outpath, 'w', encoding="UTF-8") as outfile:
            for i in reader:
                num += 1
                outfile.write(">{}\n".format(i[0]))  #
                seq = i[2]  #
                split_line = [seq[i:i + 70] for i in range(0, len(seq), 70)]
                for j in split_line:
                    outfile.write("{}\n".format(j))
                outfile.write("\n".format(j))

            outfile.write(">{}\n".format(num))  #
            seq = inp_seq  #
            split_line = [seq[i:i + 70] for i in range(0, len(seq), 70)]
            for j in split_line:
                outfile.write("{}\n".format(j))
            outfile.write("\n".format(j))
    else:
        filename = os.path.join("./data", database, "Protein_infomation.csv")
        outname = database + "_add.fasta"
        outpath = os.path.join("./data", database, outname)
        csvfile = open(filename, 'r')
        reader = csv.reader(csvfile)
        next(reader)
        num = 0
        with open(outpath, 'w', encoding="UTF-8") as outfile:
            for i in reader:
                num += 1
                outfile.write(">{}\n".format(i[0]))  #
                seq = i[1]  #
                split_line = [seq[i:i + 70] for i in range(0, len(seq), 70)]
                for j in split_line:
                    outfile.write("{}\n".format(j))
                outfile.write("\n".format(j))

            outfile.write(">{}\n".format(num))  #
            seq = inp_seq  #
            split_line = [seq[i:i + 70] for i in range(0, len(seq), 70)]
            for j in split_line:
                outfile.write("{}\n".format(j))
            outfile.write("\n".format(j))
def get_prosim_matrix(dataset):
    path = f'./data/{dataset}'
    readpath = os.path.join(path, "protein_add_sim.tsv")

    tsvfile = open(readpath, "r")
    tsvfile = tsvfile.readlines()
    tsv = pd.read_csv(readpath, sep="\t", usecols=[0], dtype={0: int})
    min_num = int(tsv.min()[0])
    max_num = int(tsv.max()[0])
    matrixnum = max_num - min_num + 1
    matrix = np.zeros((matrixnum, matrixnum))
    for i in tsvfile:
        i = i.split("\t")
        a = int(i[0].split("_")[0]) - min_num
        b = int(i[1].split("_")[0]) - min_num
        # edge = (a,b)
        matrix[a, b] = round(float(i[2]) / 100, 4)
    np.save(os.path.join(path, "SimProtein_blast.npy"), matrix)
    matrix = pd.DataFrame(matrix)
    matrix.to_csv(os.path.join(path, "SimProtein_blast_add.csv"))

def dgl_similarity_graph_d(data, args):
    drdr_matrix = k_matrix(data['sim_drug'], args.neighbor)

    prpr_matrix = k_matrix(data['sim_protein'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)

    prpr_nx = nx.from_numpy_matrix(prpr_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)

    prpr_graph = dgl.from_networkx(prpr_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['sim_drug'][:,:-1])

    prpr_graph.ndata['drs'] = torch.tensor(data['sim_protein'])

    return drdr_graph, prpr_graph, data

def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['sim_drug'], args.neighbor)

    prpr_matrix = k_matrix(data['sim_protein'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)

    prpr_nx = nx.from_numpy_matrix(prpr_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)

    prpr_graph = dgl.from_networkx(prpr_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['sim_drug'])

    prpr_graph.ndata['drs'] = torch.tensor(data['sim_protein'])

    return drdr_graph, prpr_graph, data




def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def dgl_similarity_graph_p(data, args):
    drdr_matrix = k_matrix(data['sim_drug'], args.neighbor)

    prpr_matrix = k_matrix(data['sim_protein'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)

    prpr_nx = nx.from_numpy_matrix(prpr_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)

    prpr_graph = dgl.from_networkx(prpr_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['sim_drug'])

    prpr_graph.ndata['drs'] = torch.tensor(data['sim_protein'][:,:-1])

    return drdr_graph, prpr_graph, data

if __name__ == '__main__':
    get_prosim_matrix('DAVIS')