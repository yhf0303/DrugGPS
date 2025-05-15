import torch

import torch.nn as nn
from scripts.layers import GraphTransformer_dru,GraphTransformer_dis,Encoder



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Multi_Model5(nn.Module):
    '''

    获取药物特征：morgan指纹、MACCS、Avalon、Graph2vec
    获取疾病特征：Mesh
    获取蛋白特征：ESM-2
    融合药物相似性图与蛋白相似性图
    需要进行batch——size处理


    '''
    def __init__(self,  args,drug_in_dim,target_in_dim,embed_size, num_hiddens,num_layers,num_heads,
                 labels,gt_layer,gt_head,gt_out_dim,dropout,num_drugs,num_targets,**kwargs):
        super(Multi_Model5, self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_drugs = num_drugs
        # self.num_diseases = num_diseases
        self.num_targets=num_targets

        self.activation=nn.ReLU()
        # 使用预训练的词向量作为参数



        self.linear1=nn.Linear(drug_in_dim,embed_size)

        self.linear2 = nn.Linear(target_in_dim, embed_size)

        self.drug_gat = GraphTransformer_dru(device, gt_layer, num_drugs, gt_out_dim, gt_out_dim,
                                             gt_head, dropout)
        self.target_gat = GraphTransformer_dis(device, gt_layer, num_targets, gt_out_dim, gt_out_dim,
                                            gt_head, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.target_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens * 2, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, labels))  # 全连接层
    def forward(self, drug_feats,target_feats,drug_sim_graphs,target_sim_graphs,walks_embedding,ddi):
        drug_feats=self.linear1(drug_feats)


        dis_feats=self.linear2(target_feats)

        drug_gnns = self.drug_gat(drug_sim_graphs)
        dis_gnns = self.target_gat(target_sim_graphs)

        dr = torch.stack((drug_feats,drug_gnns), dim=1)  # 663,2,200
        pro= torch.stack((dis_feats,dis_gnns), dim=1)  # 409,2,200


        dr = self.drug_trans(dr)  # 663,2,200
        pro = self.target_trans(pro)

        dr = dr.view(-1, 2 * self.num_hiddens)
        pro = pro.view(-1, 2 * self.num_hiddens)

        drdi_embedding = torch.mul(dr[ddi[:, 0]], pro[ddi[:, 1]])  # 4051,400
        # drdi_embedding = dr[ddi[:, 0]] + pro[ddi[:, 1]]

        outputs = self.mlp(drdi_embedding)
        return outputs




class Multi_Model5_plus(nn.Module):
    '''
    deepwalk模块+药物疾病相似性矩阵+特征
    获取药物特征：morgan指纹、MACCS、Avalon、Graph2vec
    获取疾病特征：Mesh
    获取蛋白特征：ESM-2

    需要进行batch——size处理

    target:disease proteins


    '''
    def __init__(self, args,drug_in_dim,target_in_dim,embed_size, num_hiddens,num_layers,num_heads,
                 labels,gt_layer,gt_head,gt_out_dim,dropout,num_drugs,num_targets,**kwargs):
        super(Multi_Model5_plus, self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_drugs = num_drugs
        self.num_targets = num_targets

        self.activation=nn.ReLU()
        # 使用预训练的词向量作为参数
        # self.embedding = nn.Embedding.from_pretrained(weight)

        self.linear1 = nn.Linear(drug_in_dim, embed_size)

        self.linear2 = nn.Linear(target_in_dim, embed_size)
        self.linear3 = nn.Linear(128, embed_size)
        self.linear4 = nn.Linear(128, embed_size)

        self.drug_gat = GraphTransformer_dru(device, gt_layer, num_drugs, gt_out_dim, gt_out_dim,
                                             gt_head, dropout)

        self.target_gat = GraphTransformer_dis(device, gt_layer, num_targets, gt_out_dim, gt_out_dim,
                                            gt_head, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.target_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens * 3, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, labels))  # 全连接层

        self.reg = nn.Linear(4, 2)
    def forward(self, drug_feats,target_feats,drug_sim_graphs,target_sim_graphs,walks_embedding,ddi):
        drug_feats = self.linear1(drug_feats)

        pro_feats = self.linear2(target_feats)

        drug_embed = walks_embedding[:self.num_drugs]
        drug_embed = self.linear3(drug_embed)

        pro_embed = walks_embedding[self.num_drugs:self.num_drugs+self.num_targets]
        pro_embed = self.linear4(pro_embed)
        drug_gnns = self.drug_gat(drug_sim_graphs)

        pro_gnns = self.target_gat(target_sim_graphs)

        dr = torch.stack((drug_feats, drug_gnns, drug_embed), dim=1)  # 663,3,200
        pr = torch.stack((pro_feats, pro_gnns, pro_embed), dim=1)  # 409,3,200

        dr = self.drug_trans(dr)  # 663,2,200
        pr = self.target_trans(pr)

        dr = dr.view(-1, 3 * self.num_hiddens)
        pr = pr.view(-1, 3 * self.num_hiddens)

        # early_x = torch.cat((d_x[ddi[:, 0]], dis_x[ddi[:, 1]]), 1)
        # early_x = self.fc_layers(early_x)
        drdi_embedding = torch.mul(dr[ddi[:, 0]], pr[ddi[:, 1]])  # 4051,400
        # drdi_embedding = dr[ddi[:, 0]] + pr[ddi[:, 1]]
        outputs = self.mlp(drdi_embedding)
        # x=early_x+outputs
        return outputs

