import argparse
import tensorflow as tf
from trainer import trainer
from data_utils import *
from model import RGCN
import torch.nn as nn
import logging
logging.basicConfig(filename='results.log', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pre', type=int, default=12)
parser.add_argument('--mn_pre', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--embedding_dim', type=int, default=2)
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--hid_dim', type=int, default=120)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--top_percent', type=float, default=0.03)
parser.add_argument('--save_path', type=str, default='output')
parser.add_argument('--load_path', type=str, default='1m3h/output')

args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#clients = [2, 0, 15, 18, 10, 5, 17]
#data_radio = [[0, 0.6, 0.2, 0.2], [0, 0.6, 0.2, 0.2], [0.02232, 0.57768, 0.2, 0.2], [0.04536, 0.55464, 0.2, 0.2], [0.28704, 0.31296, 0.2, 0.2], [0.41976, 0.18024, 0.2, 0.2], [0.48744, 0.11256, 0.2, 0.2]]
#clients = [2, 18,19, 10, 5, 17]
#data_radio = [[0, 0.6, 0.2, 0.2], [0, 0.6, 0.2, 0.2], [0, 0.6, 0.2, 0.2], [0.40704, 0.19296, 0.2, 0.2], [0.43976, 0.16024, 0.2, 0.2], [0.48744, 0.11256, 0.2, 0.2]]
#clients = [2,  18, 19, 10, 5, 17]
#data_radio = [[0, 0.6, 0.2, 0.2], [0, 0.6, 0.2, 0.2], [0, 0.6, 0.2, 0.2], [0.40704, 0.19296, 0.2, 0.2], [0.43976, 0.16024, 0.2, 0.2], [0.48744, 0.11256, 0.2, 0.2]]
#clients = [2, 0]
#data_radio = [[0.48744, 0.11256, 0.2, 0.2], [0.48744, 0.11256, 0.2, 0.2]]
#clients = [17]
#data_radio = [[0.48744, 0.11256, 0.2, 0.2]]
#clients = [10]
#data_radio = [[0.28704, 0.31296, 0.2, 0.2]]

# clients = [17]
# data_radio = [[0.48744, 0.11256, 0.2, 0.2]]
# milan
clients = [1,2,3,4,5,6,7]
data_radio = [[0, 0.6, 0.2, 0.2], [0, 0.6, 0.2, 0.2], [0.02232, 0.57768, 0.2, 0.2], [0.04536, 0.55464, 0.2, 0.2], [0.28704, 0.31296, 0.2, 0.2], [0.41976, 0.18024, 0.2, 0.2], [0.48744, 0.11256, 0.2, 0.2]]
df = []
adjs = []
node_list = []
for client_id, client in enumerate(clients):
    file_path = f'df_{client}'
    adj_path = f'adj_{client}'
    df_x = data_gen(file_path, data_radio[client_id], args.n_his, args.mn_pre, day_slot=96, batch_size=args.batch_size)
    df.append(df_x)
    adj_x = gen_adj(adj_path)
    adjs.append(adj_x)
    node_list.append(len(adj_x))
    print(f'client {client_id} is finished')


# model
model = RGCN(args).to(args.device)
model.train()
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
Trainer = trainer(model=model, data=df, adjs=adjs, nodes_list=node_list, clients=clients, args=args)

Trainer.train()
Trainer.test(model)
# print(df[0]['mean'])
# print(df[1]['mean'])
# inputs_placeholder = tf.placeholder(tf.float32, [None, args.n_his + 2*args.n_pre, args.n_route, 1], name='data_input')
# print()
# model = GSTGCN(inputs=inputs_placeholder, blocks=blocks, args=args, len_train=len(df_train), std=df_std, mean=df_mean)

# model.train(df_train, df_val)
# model.test(df_test, './output/')




