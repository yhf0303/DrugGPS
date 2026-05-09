import argparse
import time
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim.lr_scheduler as lr_scheduler
from scripts.models import Multi_Model5,Multi_Model5_plus
from scripts.datautils import *


def parser_config():
    parser = argparse.ArgumentParser()
    # 数据基础参数
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--task_tag', required=False, default='scripts')
    parser.add_argument('--device', required=False, default=torch.device('cuda:0'))
    parser.add_argument('--data_shuffle', required=False, default=True)
    parser.add_argument('--predict_tag', required=False, default="dr_di", choices = ["dr_di","di_pr","dr_pr"],help='prediction')
    parser.add_argument('--model', required=False, default="model5+", choices=["model5", "model5+"],
                        help='prediction')
    # parser.add_argument('--anneal_rate', required=False, default=0.9)

    # 超参
    parser.add_argument('--k_fold', type=int, default=1, help='k-fold cross validation')
    parser.add_argument('--batch_size', type=int, default=5, help='number of batch_size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=6, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')

    # 模型参数
    # deepwalk
    parser.add_argument('--cutoff', default=0.4, type=float, help='DeepWalk cutoff')
    parser.add_argument('--tp_factor', default=0.4, type=float, help='DeepWalk tp_factor')
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
    args.data_dir = './data/' + args.dataset + '/'
    args.result_dir = './result/' + args.dataset + '/'+args.task_tag+'/'
    args.saved_dir = './checkpoint/' + args.dataset + '/'
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)

    return args

def train_DTI(data, args):
    cross_entropy = nn.CrossEntropyLoss()
    AUCs, AUPRs ,ACCUR ,F1,MCC= [], [],[],[],[]


    for i in range(args.k_fold):
        print("------------------this is %dth cross validation------------------" % (i))
        print('Epoch\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

        if args.model=="model5+":
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
        model = model.to(args.device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        # scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(args.device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(args.device)
        X_test = torch.LongTensor(data['X_test'][i]).to(args.device)
        Y_test = data['Y_test'][i].flatten()

        # drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        # drdipr_graph = drdipr_graph.to(args.device)

        for epoch in range(args.epochs):
            model.train()
            train_preds = model(data['dr_feature'], data['pr_feature'],
                                data['drug_sim_graphs'],
                                data['pro_sim_graphs'],
                                data['DW_embedding'],
                                X_train)
            # train_preds = scripts(data['dr_feature'],  data['pr_feature'], data['DW_embedding'],X_train)
            train_loss = cross_entropy(train_preds, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_preds = model(data['dr_feature'], data['pr_feature'], data['drug_sim_graphs'],data['pro_sim_graphs'],data['DW_embedding'],X_test)

            test_prob = fn.softmax(test_preds, dim=-1)[:, 1].cpu().numpy()
            test_score = torch.argmax(test_preds, dim=-1).cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            show = [epoch + 1, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                torch.save(model.state_dict(), os.path.join(args.saved_dir, 'best.pth'))
                savedata = {'y_true': Y_test.flatten(), 'y_pred': test_score, 'y_prob': test_prob}
                df = pd.DataFrame(savedata)
                df.to_csv(f'./ROC_data/{args.dataset}_best_{str(best_auc)}.csv', index=False)
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        ACCUR.append(best_accuracy)
        F1.append(best_f1)
        MCC.append(best_mcc)

    return AUCs, AUPRs,ACCUR,F1,MCC

def train_DDI(data, args):
    cross_entropy = nn.CrossEntropyLoss()
    AUCs, AUPRs ,ACCUR ,F1,MCC= [], [],[],[],[]


    for i in range(args.k_fold):
        print("------------------this is %dth cross validation------------------" % (i))
        print('Epoch\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

        if args.model=="model5+":
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
        model = model.to(args.device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        # scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(args.device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(args.device)
        X_test = torch.LongTensor(data['X_test'][i]).to(args.device)
        Y_test = data['Y_test'][i].flatten()

        # drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        # drdipr_graph = drdipr_graph.to(args.device)

        for epoch in range(args.epochs):
            model.train()
            train_preds = model(data['dr_feature'], data['di_feature'],
                                data['drug_sim_graphs'],
                                data['dis_sim_graphs'],
                                data['DW_embedding'],
                                X_train)
            # train_preds = scripts(data['dr_feature'],  data['pr_feature'], data['DW_embedding'],X_train)
            train_loss = cross_entropy(train_preds, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_preds = model(data['dr_feature'], data['di_feature'], data['drug_sim_graphs'],
                                data['dis_sim_graphs'],
                               data['DW_embedding'],X_test)
                # test_preds = scripts(data['dr_feature'], data['pr_feature'], data['DW_embedding'],
                #                    X_test)

            test_prob = fn.softmax(test_preds, dim=-1)[:, 1].cpu().numpy()
            test_score = torch.argmax(test_preds, dim=-1).cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            show = [epoch + 1, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                torch.save(model.state_dict(), os.path.join(args.saved_dir, 'best.pth'))
                savedata = {'y_true': Y_test.flatten(), 'y_pred': test_score, 'y_prob': test_prob}
                df = pd.DataFrame(savedata)
                if float(best_auc)>0.968:
                    df.to_csv(f'./ROC_data/{args.dataset}_best_{str(best_auc)}.csv', index=False)
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        ACCUR.append(best_accuracy)
        F1.append(best_f1)
        MCC.append(best_mcc)
    return AUCs, AUPRs,ACCUR,F1,MCC



if __name__ == '__main__':
    args = parser_config()


    print("--------------------- Complete Data Loading ---------------------")

    print('Dataset:', args.dataset)
    # for i in range(args.k_fold):
    #     xtrain = data['X_train'][i]
    #     ytrain = data['Y_train'][i]
    #
    #     relation_train = []
    #     for k, lab in enumerate(ytrain):
    #         if lab == 1:
    #             relation_train.append(xtrain[k])
    #     dr_di = np.array(relation_train)
    #     dr_di[:, 1] = dr_di[:, 1] + args.drug_number
    #     dr_dr = data['dr_dr']
    #     di_di = data['di_di'] + args.drug_number
    #     dr_pr = pd.read_csv(f'./data/{args.dataset}/DrugProteinCorrelation.csv', header=0).values
    #     di_pr=pd.read_csv(f'./data/{args.dataset}/DiseaseProteinCorrelation.csv', header=0).values
    #
    #
    #     train_relation = np.concatenate([dr_pr,di_pr,dr_di,dr_dr,di_di], axis=0)
    #     df = pd.DataFrame(train_relation)
    #     df = df.sample(frac=1.0)
    #     df.to_csv(f'./data/{args.dataset}/Allrelation_4train.txt', sep='\t', header=None, index=False)



    DW_embedding = pd.read_csv(args.data_dir + 'AllEmbedding_DeepWalk_00.txt', sep=' ', header=None)
    DW_embedding = DW_embedding.sort_values(0, ascending=True)
    DW_embedding.drop(DW_embedding.columns[0], axis=1, inplace=True)

    start = time.time()
    if args.predict_tag == "dr_di":
        data, args = load_DDIdata(args)
        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)
        print("-------------------------- Start Train --------------------------")
        AUCs, AUPRs, acc, f1, mcc = train_DDI(data, args)

    else:
        data, args = load_DTIdata(args)
        data['DW_embedding'] = torch.tensor(DW_embedding.values, dtype=torch.float).to(args.device)
        print("-------------------------- Start Train --------------------------")
        AUCs, AUPRs,acc,f1,mcc = train_DTI(data, args)

    end = time.time()
    time = end - start
    print("--------------------------- End Train ---------------------------")

    print('Time:', time)
    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print('Accuracy:', acc)
    print('F1 score:', f1)
    print('MCC:', mcc)




