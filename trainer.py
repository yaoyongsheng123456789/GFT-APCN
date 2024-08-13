import torch
import torch.nn as nn
from data_utils import *
import time
import copy
from math_utils import *
import logging

class trainer(object):
    def __init__(self, model, data, adjs, nodes_list, clients, args):
        super(trainer, self).__init__()
        self.model = model
        self.data = data
        self.adjs = adjs
        self.nodes_list = nodes_list
        self.clients = clients
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # optimizer
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)

    def loss_f(self, pre, true, std):
        loss_func = torch.nn.L1Loss().to(self.args.device)
        return loss_func(pre*std, true*std)

    def train_epoch(self, model, epoch,client_id, node_num, lr):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.args.lr)
        print(f'epoch {epoch} client {client_id} start training >>>>>')
        s_time = time.time()
        H_local = []
        for j, df_batch in enumerate(self.data[client_id]['train']):
            optimizer.zero_grad()
            # print(df_batch)
            adj = self.TensorFloat(self.adjs[client_id])

            ouput = model(df_batch[:, :self.args.n_his, :, :], adj, node_num)
                
            loss = self.loss_f(ouput, df_batch[:, self.args.n_his:, :, :], self.data[client_id]['std'])
            loss.backward()
            optimizer.step()
            if j%50 == 0:
                print(f'epoch {epoch} client {client_id} train loss {loss.item()}')
        #out = torch.cat(out_sum,dim =0)#m,t,n,c
        print(f'{epoch} client{client_id} training finished. training time {time.time()-s_time :.3f}s')
#         self.val_epoch(model=model, client_id=client_id)
        return model.state_dict()

    def train(self):
#         重新训练
        
        global_model = copy.deepcopy(self.model)
        lr = self.args.lr
        lr = self.args.lr * 0.1
        check_point = torch.load(self.args.load_path)
        state_dict = check_point['state_dict']
        global_model.load_state_dict(state_dict)
        
        

        
        
        
        for epoch in range(1, self.args.epoch+1):
            w_global = []
            #H_global= []
            for client_id in range(len(self.clients)):
                w_local = self.train_epoch(copy.deepcopy(global_model).to(self.args.device), epoch, client_id, self.nodes_list[client_id], lr=lr)
                w_global.append(w_local)
                #H_global.append(out)S
                
            w_global_avg = FedAvg(w_global)
            global_model.load_state_dict(w_global_avg)
            # if epoch % 5 == 0:
            #     lr = lr * 0.8

        self.save_model(global_model)

    def save_model(self, model):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args
        }
        torch.save(state, self.args.save_path)
        print(f'save model to {self.args.save_path}')

    def val_epoch(self, model, client_id):
        y_pred_local = []
        y_true_local = []
        for batch_idx, df_batch in enumerate(self.data[client_id]['val']):
            # transformer tensor
            adj = self.TensorFloat(self.adjs[client_id])

            test_x = df_batch[:, :self.args.n_his, :, :]
            test_y = df_batch[:, self.args.n_his:self.args.n_his+1, :, :]
            output = model(test_x, adj, self.nodes_list[client_id])
            y_true_local.append(test_y)
            y_pred_local.append(output)

        y_true_local = torch.cat(y_true_local, dim=0)
        y_pred_local = torch.cat(y_pred_local, dim=0)
        # metrics
        metrics = evaluation(y_true_local.cpu().numpy(), y_pred_local.cpu().numpy(),
                             mean=self.data[client_id]['mean'], std=self.data[client_id]['std'], n_pre=self.args.n_pre)
        # log

        mean_metrics = np.mean(metrics, axis=0)
        print(f'val client{client_id} mean MAE:{mean_metrics[0]:.4f} RMSE:{mean_metrics[1]:.4f} MAPE:{mean_metrics[2]:.4f}')

    def test(self, model):
        # load model
        check_point = torch.load(self.args.save_path)
        state_dict = check_point['state_dict']
        args = check_point['args']
        model.load_state_dict(state_dict)
        model.to(args.device)

        model.eval()
        y_pred_global = []
        y_true_global = []
        H_global = []
        with torch.no_grad():
            for client_id in range(len(self.clients)):
                y_pred_local = []
                y_true_local = []
                out_sum =[] 
                for batch_idx, df_batch in enumerate(self.data[client_id]['test']):
                    # transformer tensor
                    adj = self.TensorFloat(self.adjs[client_id])
                    test_y = df_batch[:, -self.args.mn_pre:, :, :]
                    test_x = df_batch[:, :self.args.n_his, :, :]
                    y_true_local.append(test_y)
                                
                    output = model(test_x, adj, self.nodes_list[client_id]) 
                    
                    # print(test_x.shape)
                    y_pred_local.append(output)
                
                y_true_local = torch.cat(y_true_local, dim=0)
                y_pred_local = torch.cat(y_pred_local, dim=0)
                # metrics
                metrics = evaluation(y_true_local.cpu().numpy(), y_pred_local.cpu().numpy(),
                                     mean=self.data[client_id]['mean'], std=self.data[client_id]['std'], n_pre=self.args.mn_pre)
                # log
                for i in range(self.args.mn_pre):
                    print(f'test client{client_id} time_step{i} MAE:{metrics[i][0]:.4f} RMSE:{metrics[i][1]:.4f} MAPE:{metrics[i][2]:.4f}')
                    logging.info(f'test client{client_id} time_step{i} MAE:{metrics[i][0]:.4f} RMSE:{metrics[i][1]:.4f} MAPE:{metrics[i][2]:.4f}')

                mean_metrics = np.mean(metrics, axis=0)
                print(f'client{client_id} mean MAE:{mean_metrics[0]:.4f} RMSE:{mean_metrics[1]:.4f} MAPE:{mean_metrics[2]:.4f}')
                logging.info(f'client{client_id} mean MAE:{mean_metrics[0]:.4f} RMSE:{mean_metrics[1]:.4f} MAPE:{mean_metrics[2]:.4f}')
                y_pred_global.append(y_pred_local*self.data[client_id]['std'] + self.data[client_id]['mean'])
                y_true_global.append(y_true_local*self.data[client_id]['std'] + self.data[client_id]['mean'])
            

            
            # global metrics
            y_pred_global = torch.cat(y_pred_global, dim=2)
            y_true_global = torch.cat(y_true_global, dim=2)
            metrics = evaluation(y_true_global.cpu().numpy(), y_pred_global.cpu().numpy(),
                                 mean=0, std=1,
                                 n_pre=self.args.mn_pre)
            # log
            for i in range(self.args.mn_pre):
                print(
                    f'test total time_step{i} MAE:{metrics[i][0]:.4f} RMSE:{metrics[i][1]:.4f} MAPE:{metrics[i][2]:.4f}')
                logging.info(f'test total time_step{i} MAE:{metrics[i][0]:.4f} RMSE:{metrics[i][1]:.4f} MAPE:{metrics[i][2]:.4f}')

            mean_metrics = np.mean(metrics, axis=0)
            print(
                f'total mean MAE:{mean_metrics[0]:.4f} RMSE:{mean_metrics[1]:.4f} MAPE:{mean_metrics[2]:.4f}')
            logging.info( f'total mean MAE:{mean_metrics[0]:.4f} RMSE:{mean_metrics[1]:.4f} MAPE:{mean_metrics[2]:.4f}')
