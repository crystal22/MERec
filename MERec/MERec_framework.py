#!/usr/bin/env python3

"""
Trains a LSTM with MAML on four cities Dataset.

"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn, optim
from maml import MAML
import numpy
from metrics import Metrics
from itertools import cycle

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print(out.shape)
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        out = self.fc(out[:, -1, :])

        h_n_last = hn[-1]

        # print(out.shape)
        # out.size() --> 100, 10
        return out, h_n_last

def main(
        shots=10,
        tasks_per_batch=16,
        num_tasks=4,
        adapt_lr=0.01,
        meta_lr=0.01,
        ft_lr = 0.01,
        poi_lr = 0.001,
        adapt_steps=5,
        input_dim = 150,
        input_dim_poi = 200,
        hidden_dim=32,
        layer_dim = 3,
        output_dim_poi_level = 1085,
        output_dim_category_level = 1085,
        tasks_length = 4,
        epchos = 500,
        epcho_ft = 1000,
        batch_size = 64

):
    # load the dataset
    train_cal = torch.load('./Dataset/PoiData/cal_dl_train_poi')
    test_cal = torch.load('./Dataset/PoiData/cal_dl_test_poi')

    train_pho = torch.load('./Dataset/PoiData/pho_dl_train_poi')
    test_pho = torch.load('./Dataset/PoiData/pho_dl_test_poi')

    train_sin = torch.load('./Dataset/PoiData/sin_dl_train_poi')
    test_sin = torch.load('./Dataset/PoiData/sin_dl_test_poi')

    train_newy = torch.load('./Dataset/PoiData/newy_dl_train_poi')
    test_newy = torch.load('./Dataset/PoiData/newy_dl_test_poi')

    dl_poi_train_cal = DataLoader(train_cal, batch_size=batch_size, shuffle=True, drop_last= True)
    dl_poi_test_cal = DataLoader(test_cal, batch_size=batch_size, shuffle=True, drop_last= True)

    dl_poi_train_pho = DataLoader(train_pho, batch_size=batch_size, shuffle=True, drop_last= True)
    dl_poi_test_pho = DataLoader(test_pho, batch_size=batch_size, shuffle=True, drop_last= True)

    dl_poi_train_sin = DataLoader(train_sin, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_poi_test_sin = DataLoader(test_sin, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_poi_train_newy = DataLoader(train_newy, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_poi_test_newy = DataLoader(test_newy, batch_size=batch_size, shuffle=True, drop_last=True)

    ############################################################################################
    #Meta-learning dataset
    qry_cal = torch.load('./Dataset/MetaData/cal_dl_qry')
    sup_cal = torch.load('./Dataset/MetaData/cal_dl_sup')

    qry_sin = torch.load('./Dataset/MetaData/sin_dl_qry')
    sup_sin = torch.load('./Dataset/MetaData/sin_dl_sup')

    qry_pho = torch.load('./Dataset/MetaData/pho_dl_qry')
    sup_pho = torch.load('./Dataset/MetaData/pho_dl_sup')

    qry_newy = torch.load('./Dataset/MetaData/newy_dl_qry')
    sup_newy = torch.load('./Dataset/MetaData/newy_dl_sup')

    dl_sup_cal = DataLoader(qry_cal, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_qry_cal = DataLoader(sup_cal, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_sup_pho = DataLoader(qry_pho, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_qry_pho = DataLoader(sup_pho, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_sup_sin = DataLoader(qry_sin, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_qry_sin = DataLoader(sup_sin, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_sup_newy = DataLoader(qry_newy, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_qry_newy = DataLoader(sup_newy, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_sup_c1,dl_qry_c1 = dl_sup_cal,dl_qry_cal
    dl_sup_c2, dl_qry_c2 = dl_sup_pho, dl_qry_pho
    dl_sup_c3, dl_qry_c3 = dl_sup_sin, dl_qry_sin
    dl_sup_c4, dl_qry_c4 = dl_sup_newy, dl_qry_newy
    ########################################################################################################
    ft_train_cal = torch.load('./Dataset/FinetuneData/cal_dl_train_ft')

    ft_train_sin = torch.load('./Dataset/FinetuneData/sin_dl_train_ft')

    ft_train_pho = torch.load('./Dataset/FinetuneData/pho_dl_train_ft')

    ft_train_newy = torch.load('./Dataset/FinetuneData/newy_dl_train_ft')

    dl_ft_train_cal = DataLoader(ft_train_cal, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_ft_train_sin = DataLoader(ft_train_sin, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_ft_train_pho = DataLoader(ft_train_pho, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_ft_train_newy = DataLoader(ft_train_newy, batch_size=batch_size, shuffle=True, drop_last=True)

    ###
    ########################################################################################################
    #for step, (batchX, batchY) in enumerate(dl_sup_cal):
        #print('| Step: ', step, '| batch x: ',
              #len(batchX.numpy()), '| batch y: ', len(batchY.numpy()))
        #print(batchX.shape, batchY.shape)

    ###################################################
    corelation_strategy_cal = [1,1,0.87,0.81,0.64]
    print("Strat>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # create the model
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_poi_level)
    maml = MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    print(maml)
    opt = optim.Adam(maml.parameters(), meta_lr)
    #lossfn = nn.MSELoss(reduction='mean')
    lossfn = nn.CrossEntropyLoss()

    #cal_train_data = zip(dl_sup_cal, dl_qry_cal)
    #pho_train_data = zip(dl_sup_pho, dl_qry_pho)
    #tasks = zip(zip(dl_sup_cal, dl_qry_cal), zip(dl_sup_pho, dl_qry_pho))
    print('LSTM parameters: ')
    for name, param in model.named_parameters():
        print(name)

    print('MAML parameters: ')
    for name, param in maml.named_parameters():
        print(name, param)
        break

    """
    Meta-learning with lstm Train Progress:
    """
    for epcho in range(epchos):
        tasks = zip(zip(dl_sup_c1, dl_qry_c1), zip(dl_sup_c2, dl_qry_c2), zip(dl_sup_c3, dl_qry_c3), zip(dl_sup_c4, dl_qry_c4))
        #print ('Epcho: ', epcho)
        #retrive data from four tasks
        meta_loss=[]
        n = 1
        for t, iteration_batch_task in enumerate(tasks):
            meta_train_loss = 0.0
            for i, (support, query) in enumerate(iteration_batch_task):

                learner = maml.clone()

                for _ in range(adapt_steps):  # adaptation_steps
                    support_preds, hl_support = learner(support[0])
                    #print(hl_support)
                    support_loss = lossfn(support_preds, support[1].long())
                    learner.adapt(support_loss)
                    #print('support_loss', support_loss)

                query_preds, hl_query = learner(query[0])
                query_loss = lossfn(query_preds, query[1].long())
                #Correlation Strategy
                query_loss = query_loss*corelation_strategy_cal[t]
                meta_train_loss += query_loss
                #print('query_loss', query_loss)
            meta_train_loss = (meta_train_loss / tasks_length)
            #print('meta_train_loss', meta_train_loss)
            meta_loss.append(meta_train_loss.item())
            #if epcho % 10 == 0:
                #print('Epcho:', epcho, 'Meta Train Loss: ', meta_train_loss.item())

            opt.zero_grad()
            meta_train_loss.backward()
            opt.step()

        print('Epcho: ', epcho, 'Meta-average-loss: ', numpy.mean(meta_loss))

        #########################################################################

    print('LSTM parameters after training: ')
    for name, param in learner.named_parameters():
        print(name, param)
        break

    print('MAML parameters after training: ')
    for name, param in maml.named_parameters():
        print(name, param)
        break


    """ 
    Fine Tune Stage 
    """
    model_ft = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_category_level)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=ft_lr)
    #lossfn_ft = nn.CrossEntropyLoss()

    print('Model_ft parameters_before copy: ')
    for name, param in model_ft.named_parameters():
        print(name, param)
        break

    print('COPY>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # original saved file with DataParallel
    state_dict = maml.state_dict()  #
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    # load params
    model_ft.load_state_dict(new_state_dict)  #


    print('Model_ft parameters_after copy: ')
    for name, param in model_ft.named_parameters():
        print(name, param)
        break

    """ 
        POI level Stage 
    """

    ###
    model_poi = LSTMModel(input_dim_poi, hidden_dim, layer_dim, output_dim_poi_level)
    optimizer_poi = optim.Adam(model_poi.parameters(), lr=poi_lr)
    lossfn_poi = nn.CrossEntropyLoss()

    total_loss = []
    for epcho in range(epcho_ft):
        loss_all=[]
        for i, data in enumerate(zip(dl_ft_train_cal, dl_poi_train_cal)):

            train_x_ft = data[0][0]
            #print(train_x_ft.shape)
            label_ft = data[0][1]

            train_x_poi = data[1][0]
            #print(train_x_poi.shape)
            label_poi = data[1][1]

            optimizer_ft.zero_grad()
            optimizer_poi.zero_grad()

            pred_category, hl_category = model_ft(train_x_ft)
            #print("hl_category: ", hl_category.shape)
            pre_poi, hl_poi = model_poi(train_x_poi)

            conca = torch.cat([hl_category,hl_poi],1)
            #print("Conca: ------------",conca.shape[1])
            m = torch.nn.Linear(conca.shape[1],output_dim_poi_level)
            output = m(conca)

            loss0 = lossfn_poi(output, label_poi.long())
            loss1 = lossfn_poi(output, label_poi.long())

            loss = (loss0 + loss1)/2

            loss.backward()

            optimizer_ft.step()
            optimizer_poi.step()

            loss_all.append(loss1.item())
        l = numpy.mean(loss_all)

        if epcho %20 ==0:
            print('Epcho: ', epcho, 'Fine_Tune_Average_Loss: ', l)
            total_loss.append(l)
            metric = Metrics()

            hit5_socre_all = []
            hit10_socre_all = []
            ndcg5_socre_all = []
            ndcg10_socre_all = []

            for i, (input, label) in enumerate(dl_poi_test_cal):
                pred, hl_test = model_poi(input)
                """
                Hit@5 and Hit@10
                """
                hit_socre5 = metric.hits_score(label, pred, k =5)
                hit_socre10 = metric.hits_score(label, pred, k=10)
                hit5_socre_all.append(hit_socre5)
                hit10_socre_all.append(hit_socre10)

                """
                nDCG@5 and nDCG@10
                """
                ndcg_socre5 = metric.ndcg_score(label, pred, k=5)
                ndcg_socre10 = metric.ndcg_score(label, pred, k=10)
                ndcg5_socre_all.append(ndcg_socre5)
                ndcg10_socre_all.append(ndcg_socre10)

            print('Hit@5: ',numpy.mean(hit5_socre_all),'Hit@10: ', numpy.mean(hit10_socre_all))
            print('nDCG@5: ', numpy.mean(ndcg5_socre_all), 'nDCG@10: ', numpy.mean(ndcg10_socre_all))


if __name__ == '__main__':
    main()

