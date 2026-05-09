import torch

import torch.nn as nn
from scripts.layers import GraphTransformer_dru,GraphTransformer_dis,Encoder
from torch.nn.utils.weight_norm import weight_norm
from scripts.mol_encoder import AtomEncoder,GINConv,GCNConv
import torch.nn.functional as F
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

from torch.nn.init import xavier_uniform_, zeros_  # 对应TensorFlow的glorot_uniform和zeros初始化


class Protein3DConvModel(nn.Module):
    def __init__(self, input_channels=16):
        """
        蛋白质3D卷积模型（PyTorch版）
        Args:
            input_channels: 输入特征的通道数（原TensorFlow模型中为16，需与特征提取输出一致）
        """
        super(Protein3DConvModel, self).__init__()

        # -------------------------- 1. 初始卷积层（对应conv1） --------------------------
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,  # 输入通道数：8（与原模型一致）
            out_channels=96,  # 输出通道数：96
            kernel_size=(3, 3, 3),  # 卷积核大小：3×3×3
            stride=(2, 2, 2),  # 步长：2×2×2（下采样）
            padding=(1,1,1)  # 填充：same（输入输出尺寸按步长比例变化）
        )
        # 初始化conv1的权重和偏置（对应TensorFlow的glorot_uniform和zeros）
        xavier_uniform_(self.conv1.weight)
        zeros_(self.conv1.bias)

        # -------------------------- 2. Fire2模块（squeeze + expand1 + expand2） --------------------------
        self.fire2_squeeze = nn.Conv3d(96, 16, kernel_size=(1, 1, 1), padding='same')
        self.fire2_expand1 = nn.Conv3d(16, 64, kernel_size=(1, 1, 1), padding='same')
        self.fire2_expand2 = nn.Conv3d(16, 64, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire2参数
        xavier_uniform_(self.fire2_squeeze.weight);
        zeros_(self.fire2_squeeze.bias)
        xavier_uniform_(self.fire2_expand1.weight);
        zeros_(self.fire2_expand1.bias)
        xavier_uniform_(self.fire2_expand2.weight);
        zeros_(self.fire2_expand2.bias)

        # -------------------------- 3. Fire3模块 --------------------------
        self.fire3_squeeze = nn.Conv3d(128, 16, kernel_size=(1, 1, 1), padding='same')
        self.fire3_expand1 = nn.Conv3d(16, 64, kernel_size=(1, 1, 1), padding='same')
        self.fire3_expand2 = nn.Conv3d(16, 64, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire3参数
        xavier_uniform_(self.fire3_squeeze.weight);
        zeros_(self.fire3_squeeze.bias)
        xavier_uniform_(self.fire3_expand1.weight);
        zeros_(self.fire3_expand1.bias)
        xavier_uniform_(self.fire3_expand2.weight);
        zeros_(self.fire3_expand2.bias)

        # -------------------------- 4. Fire4模块 --------------------------
        self.fire4_squeeze = nn.Conv3d(128, 32, kernel_size=(1, 1, 1), padding='same')
        self.fire4_expand1 = nn.Conv3d(32, 128, kernel_size=(1, 1, 1), padding='same')
        self.fire4_expand2 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire4参数
        xavier_uniform_(self.fire4_squeeze.weight);
        zeros_(self.fire4_squeeze.bias)
        xavier_uniform_(self.fire4_expand1.weight);
        zeros_(self.fire4_expand1.bias)
        xavier_uniform_(self.fire4_expand2.weight);
        zeros_(self.fire4_expand2.bias)

        # -------------------------- 5. 最大池化层（对应maxpool_4） --------------------------
        self.maxpool4 = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(3, 3, 3),
            padding=(1,1,1)
        )

        # -------------------------- 6. Fire5模块 --------------------------
        self.fire5_squeeze = nn.Conv3d(256, 32, kernel_size=(1, 1, 1), padding='same')
        self.fire5_expand1 = nn.Conv3d(32, 128, kernel_size=(1, 1, 1), padding='same')
        self.fire5_expand2 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire5参数
        xavier_uniform_(self.fire5_squeeze.weight);
        zeros_(self.fire5_squeeze.bias)
        xavier_uniform_(self.fire5_expand1.weight);
        zeros_(self.fire5_expand1.bias)
        xavier_uniform_(self.fire5_expand2.weight);
        zeros_(self.fire5_expand2.bias)

        # -------------------------- 7. Fire6模块 --------------------------
        self.fire6_squeeze = nn.Conv3d(256, 48, kernel_size=(1, 1, 1), padding='same')
        self.fire6_expand1 = nn.Conv3d(48, 192, kernel_size=(1, 1, 1), padding='same')
        self.fire6_expand2 = nn.Conv3d(48, 192, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire6参数
        xavier_uniform_(self.fire6_squeeze.weight);
        zeros_(self.fire6_squeeze.bias)
        xavier_uniform_(self.fire6_expand1.weight);
        zeros_(self.fire6_expand1.bias)
        xavier_uniform_(self.fire6_expand2.weight);
        zeros_(self.fire6_expand2.bias)

        # -------------------------- 8. Fire7模块 --------------------------
        self.fire7_squeeze = nn.Conv3d(384, 48, kernel_size=(1, 1, 1), padding='same')  # 输入384=192+192
        self.fire7_expand1 = nn.Conv3d(48, 192, kernel_size=(1, 1, 1), padding='same')
        self.fire7_expand2 = nn.Conv3d(48, 192, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire7参数
        xavier_uniform_(self.fire7_squeeze.weight);
        zeros_(self.fire7_squeeze.bias)
        xavier_uniform_(self.fire7_expand1.weight);
        zeros_(self.fire7_expand1.bias)
        xavier_uniform_(self.fire7_expand2.weight);
        zeros_(self.fire7_expand2.bias)

        # -------------------------- 9. Fire8模块 --------------------------
        self.fire8_squeeze = nn.Conv3d(384, 64, kernel_size=(1, 1, 1), padding='same')  # 输入384=192+192
        self.fire8_expand1 = nn.Conv3d(64, 256, kernel_size=(1, 1, 1), padding='same')
        self.fire8_expand2 = nn.Conv3d(64, 256, kernel_size=(3, 3, 3), padding='same')
        # 初始化Fire8参数
        xavier_uniform_(self.fire8_squeeze.weight);
        zeros_(self.fire8_squeeze.bias)
        xavier_uniform_(self.fire8_expand1.weight);
        zeros_(self.fire8_expand1.bias)
        xavier_uniform_(self.fire8_expand2.weight);
        zeros_(self.fire8_expand2.bias)

        # -------------------------- 10. 平均池化层（对应avg8） --------------------------
        self.avgpool8 = nn.AvgPool3d(
            kernel_size=(3, 3, 3),
            stride=(3, 3, 3),
            padding=(1,1,1)
        )

        # -------------------------- 11. 全连接层（对应Dense(1)） --------------------------

        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(in_features=4096, out_features=300)  #
        xavier_uniform_(self.fc_out.weight)
        zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        前向传播（输入尺寸：[batch_size, 16, 34, 34, 34]）
        PyTorch中3D卷积的输入格式为：[batch, channels, depth, height, width]
        （与TensorFlow的[batch, depth, height, width, channels]维度顺序不同）
        """
        # 1. Conv1 + ReLU
        x = F.relu(self.conv1(x))  # 输出尺寸：[batch, 96, 17, 17, 17]（34/2=17，same填充）
        # 2. Fire2：squeeze → 分支expand → 拼接
        x = F.relu(self.fire2_squeeze(x))
        exp1 = F.relu(self.fire2_expand1(x))
        exp2 = F.relu(self.fire2_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)

        # 3. Fire3（同Fire2结构）
        x = F.relu(self.fire3_squeeze(x))
        exp1 = F.relu(self.fire3_expand1(x))
        exp2 = F.relu(self.fire3_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)

        # 4. Fire4
        x = F.relu(self.fire4_squeeze(x))
        exp1 = F.relu(self.fire4_expand1(x))
        exp2 = F.relu(self.fire4_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)

        # 5. MaxPool4
        x = self.maxpool4(x)

        # 6. Fire5
        x = F.relu(self.fire5_squeeze(x))
        exp1 = F.relu(self.fire5_expand1(x))
        exp2 = F.relu(self.fire5_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)

        # 7. Fire6
        x = F.relu(self.fire6_squeeze(x))
        exp1 = F.relu(self.fire6_expand1(x))
        exp2 = F.relu(self.fire6_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)

        # 8. Fire7
        x = F.relu(self.fire7_squeeze(x))
        exp1 = F.relu(self.fire7_expand1(x))
        exp2 = F.relu(self.fire7_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)
        # 9. Fire8
        x = F.relu(self.fire8_squeeze(x))
        exp1 = F.relu(self.fire8_expand1(x))
        exp2 = F.relu(self.fire8_expand2(x))
        x = torch.cat([exp1, exp2], dim=1)
        # 10. AvgPool8
        x = self.avgpool8(x)
        # 11. Flatten + 全连接
        x = self.flatten(x)  # 展平：512 × 2×2×2 = 4096
        x = self.fc_out(x)  # 输出：[batch, 1]（线性激活，对应原模型的linear）
        return x



class GNNComplete(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNNComplete, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=1):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)   #池化窗口为3

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            #print(self.h_mat.shape,v_.shape,q_.shape)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


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

class Multi_Model5_ban(nn.Module):
    '''
    deepwalk模块+药物疾病相似性矩阵+特征
    获取药物特征：morgan指纹、MACCS、Avalon、Graph2vec
    获取疾病特征：Mesh
    获取蛋白特征：ESM-2

    需要进行batch——size处理

    target:disease proteins


    '''
    def __init__(self, args,drug_in_dim,target_in_dim,embed_size, num_hiddens,num_layers,num_heads,
                 labels,gt_layer,gt_head,gt_out_dim,dropout,num_drugs,num_targets,drug_graph,protein_graph,**kwargs):
        super(Multi_Model5_ban, self).__init__(**kwargs)

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
        self.drug_sim_graphs=drug_graph
        self.target_sim_graphs=protein_graph
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.target_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.CNN3D=Protein3DConvModel(8)
        # self.ban=BANLayer(num_hiddens,num_hiddens,200,6)
        # self.mlp = nn.Sequential(
        #     nn.Linear(200 , 3*200),
        #     self.activation,
        #     nn.Dropout(0.2),
        #     nn.Linear(3*200, 200),
        #     self.activation,
        #     nn.Dropout(0.2),
        #     nn.Linear(200, labels))  # 全连接层

        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens*2),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens*2, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, labels))  # 全连接层

        # ========== 新增代码：冻结图特征的梯度 ==========
        self._freeze_graph_features()

    def _freeze_graph_features(self):
        """
        冻结图中所有节点和边特征的梯度，防止它们被反向传播更新，
        并避免计算图污染问题。
        """
        if self.drug_sim_graphs is not None:
            # 遍历图的所有节点特征
            for feat_name in self.drug_sim_graphs.ndata:
                # detach() 会创建一个新的张量，从当前计算图中分离出来。
                # 然后将其重新赋值给图的特征。
                self.drug_sim_graphs.ndata[feat_name] = self.drug_sim_graphs.ndata[feat_name].detach()
                # 明确设置 requires_grad 为 False
                self.drug_sim_graphs.ndata[feat_name].requires_grad = False
            # 遍历图的所有边特征（如果有的话）
            for feat_name in self.drug_sim_graphs.edata:
                self.drug_sim_graphs.edata[feat_name] = self.drug_sim_graphs.edata[feat_name].detach()
                self.drug_sim_graphs.edata[feat_name].requires_grad = False

        if self.target_sim_graphs is not None:
            for feat_name in self.target_sim_graphs.ndata:
                self.target_sim_graphs.ndata[feat_name] = self.target_sim_graphs.ndata[feat_name].detach()
                self.target_sim_graphs.ndata[feat_name].requires_grad = False
            for feat_name in self.target_sim_graphs.edata:
                self.target_sim_graphs.edata[feat_name] = self.target_sim_graphs.edata[feat_name].detach()
                self.target_sim_graphs.edata[feat_name].requires_grad = False

    def forward(self, drug_feats,target_feats,dr_embedding,pr_embedding,drug_3d,protein_3d,ddi):
        drug_feats = self.linear1(drug_feats)

        pro_feats = self.linear2(target_feats)


        drug_embed = self.linear3(dr_embedding)


        pro_embed = self.linear4(pr_embedding)


        drug_graph_features = self.drug_sim_graphs.ndata['drs']
        protein_graph_features = self.target_sim_graphs.ndata['drs']  # 假设蛋白图的特征键是 'prs'

        # 传入图和特征张量
        drug_gnns = self.drug_gat(self.drug_sim_graphs, drug_graph_features)
        pro_gnns = self.target_gat(self.target_sim_graphs, protein_graph_features)




        drug_gnns1=drug_gnns[ddi[:, 0]]
        pro_gnns1=pro_gnns[ddi[:, 1]]
        protein_3d2 = protein_3d.permute(0, 4, 1, 2, 3)
        pro_3d=self.CNN3D(protein_3d2)
        drug_3d=torch.sum(drug_3d, dim=1)
        dr = torch.stack((drug_feats, drug_gnns1, drug_embed,drug_3d), dim=1)  # 663,3,200     663为药物数量

        # dr=torch.cat([dr,drug_3d],dim=1)
        pr = torch.stack((pro_feats, pro_gnns1, pro_embed,pro_3d), dim=1)  # 409,3,200       409为蛋白数量

        dr = self.drug_trans(dr)  # 663,3,200
        pr = self.target_trans(pr) # 409,3,200
        dr=torch.sum(dr, dim=1)
        pr=torch.sum(pr, dim=1)
        # dr = dr.view(-1, 4 * self.num_hiddens)
        # pr = pr.view(-1, 4 * self.num_hiddens)
        # drf=dr[ddi[:, 0]]
        # prf=pr[ddi[:, 1]]
        # fusion_feat = dr[ddi[:, 0]] + pr[ddi[:, 1]]
        fusion_feat = torch.mul(dr, pr)
        # fusion_feat,att_map=self.ban(drf,prf)  #4051*200
        outputs = self.mlp(fusion_feat)
        # x=early_x+outputs
        return outputs



class Multi_Model5_ban1(nn.Module):
    '''
    deepwalk模块+药物疾病相似性矩阵+特征
    获取药物特征：morgan指纹、MACCS、Avalon、Graph2vec
    获取疾病特征：Mesh
    获取蛋白特征：ESM-2

    需要进行batch——size处理

    target:disease proteins


    '''
    def __init__(self, args,drug_in_dim,target_in_dim,embed_size, num_hiddens,num_layers,num_heads,
                 labels,gt_layer,gt_head,gt_out_dim,dropout,num_drugs,num_targets,drug_graph,protein_graph,**kwargs):
        super(Multi_Model5_ban1, self).__init__(**kwargs)

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
        self.drug_sim_graphs=drug_graph
        self.target_sim_graphs=protein_graph
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.target_trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.ban=BANLayer(num_hiddens,num_hiddens,200,6)
        # self.mlp = nn.Sequential(
        #     nn.Linear(200 , 3*200),
        #     self.activation,
        #     nn.Dropout(0.2),
        #     nn.Linear(3*200, 200),
        #     self.activation,
        #     nn.Dropout(0.2),
        #     nn.Linear(200, labels))  # 全连接层

        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens*2),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens*2, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, labels))  # 全连接层

        # ========== 新增代码：冻结图特征的梯度 ==========
        self._freeze_graph_features()

    def _freeze_graph_features(self):
        """
        冻结图中所有节点和边特征的梯度，防止它们被反向传播更新，
        并避免计算图污染问题。
        """
        if self.drug_sim_graphs is not None:
            # 遍历图的所有节点特征
            for feat_name in self.drug_sim_graphs.ndata:
                # detach() 会创建一个新的张量，从当前计算图中分离出来。
                # 然后将其重新赋值给图的特征。
                self.drug_sim_graphs.ndata[feat_name] = self.drug_sim_graphs.ndata[feat_name].detach()
                # 明确设置 requires_grad 为 False
                self.drug_sim_graphs.ndata[feat_name].requires_grad = False
            # 遍历图的所有边特征（如果有的话）
            for feat_name in self.drug_sim_graphs.edata:
                self.drug_sim_graphs.edata[feat_name] = self.drug_sim_graphs.edata[feat_name].detach()
                self.drug_sim_graphs.edata[feat_name].requires_grad = False

        if self.target_sim_graphs is not None:
            for feat_name in self.target_sim_graphs.ndata:
                self.target_sim_graphs.ndata[feat_name] = self.target_sim_graphs.ndata[feat_name].detach()
                self.target_sim_graphs.ndata[feat_name].requires_grad = False
            for feat_name in self.target_sim_graphs.edata:
                self.target_sim_graphs.edata[feat_name] = self.target_sim_graphs.edata[feat_name].detach()
                self.target_sim_graphs.edata[feat_name].requires_grad = False

    def forward(self, drug_feats,target_feats,dr_embedding,pr_embedding,ddi):
        drug_feats = self.linear1(drug_feats)

        pro_feats = self.linear2(target_feats)


        drug_embed = self.linear3(dr_embedding)


        pro_embed = self.linear4(pr_embedding)


        drug_graph_features = self.drug_sim_graphs.ndata['drs']
        protein_graph_features = self.target_sim_graphs.ndata['drs']  # 假设蛋白图的特征键是 'prs'

        # 传入图和特征张量
        drug_gnns = self.drug_gat(self.drug_sim_graphs, drug_graph_features)
        pro_gnns = self.target_gat(self.target_sim_graphs, protein_graph_features)




        drug_gnns1=drug_gnns[ddi[:, 0]]
        pro_gnns1=pro_gnns[ddi[:, 1]]

        dr = torch.stack((drug_feats, drug_gnns1, drug_embed), dim=1)  # 663,3,200     663为药物数量

        # dr=torch.cat([dr,drug_3d],dim=1)
        pr = torch.stack((pro_feats, pro_gnns1, pro_embed), dim=1)  # 409,3,200       409为蛋白数量

        dr = self.drug_trans(dr)  # 663,3,200
        pr = self.target_trans(pr) # 409,3,200
        dr=torch.sum(dr, dim=1)
        pr=torch.sum(pr, dim=1)
        # dr = dr.view(-1, 4 * self.num_hiddens)
        # pr = pr.view(-1, 4 * self.num_hiddens)
        # drf=dr[ddi[:, 0]]
        # prf=pr[ddi[:, 1]]
        # fusion_feat = dr[ddi[:, 0]] + pr[ddi[:, 1]]
        fusion_feat = torch.mul(dr, pr)
        # fusion_feat,att_map=self.ban(drf,prf)  #4051*200
        outputs = self.mlp(fusion_feat)
        # x=early_x+outputs
        return outputs




class Multi_Model5_3d(nn.Module):
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
        super(Multi_Model5_3d, self).__init__(**kwargs)

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
        self.CNN3D=Protein3DConvModel(8)

        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens*2),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens*2, num_hiddens),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, labels))  # 全连接层

        # ========== 新增代码：冻结图特征的梯度 ==========

    def forward(self, drug_feats,target_feats,drug_sim_graphs,target_sim_graphs,walks_embedding,drug_3d,protein_3d,ddi):
        drug_feats = self.linear1(drug_feats)

        pro_feats = self.linear2(target_feats)

        drug_embed = walks_embedding[:self.num_drugs]
        drug_embed = self.linear3(drug_embed)

        pro_embed = walks_embedding[self.num_drugs:self.num_drugs + self.num_targets]
        pro_embed = self.linear4(pro_embed)


        drug_graph_features = drug_sim_graphs.ndata['drs']
        protein_graph_features = target_sim_graphs.ndata['drs']  # 假设蛋白图的特征键是 'prs'

        # 传入图和特征张量
        drug_gnns = self.drug_gat(drug_sim_graphs, drug_graph_features)
        pro_gnns = self.target_gat(target_sim_graphs, protein_graph_features)



        protein_3d2 = protein_3d.permute(0, 4, 1, 2, 3)
        pro_3d=self.CNN3D(protein_3d2)
        drug_3d=torch.sum(drug_3d, dim=1)
        dr = torch.stack((drug_feats, drug_gnns, drug_embed,drug_3d), dim=1)  # 663,3,200     663为药物数量

        # dr=torch.cat([dr,drug_3d],dim=1)
        pr = torch.stack((pro_feats, pro_gnns, pro_embed,pro_3d), dim=1)  # 409,3,200       409为蛋白数量

        dr = self.drug_trans(dr)  # 663,3,200
        pr = self.target_trans(pr) # 409,3,200
        dr=torch.sum(dr, dim=1)
        pr=torch.sum(pr, dim=1)
        # dr = dr.view(-1, 4 * self.num_hiddens)
        # pr = pr.view(-1, 4 * self.num_hiddens)
        # drf=dr[ddi[:, 0]]
        # prf=pr[ddi[:, 1]]
        # fusion_feat = dr[ddi[:, 0]] + pr[ddi[:, 1]]
        fusion_feat = torch.mul(dr[ddi[:, 0]], pr[ddi[:, 1]])
        # fusion_feat,att_map=self.ban(drf,prf)  #4051*200
        outputs = self.mlp(fusion_feat)
        # x=early_x+outputs
        return outputs