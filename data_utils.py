import os
import numpy as np
from math_utils import *
import pandas as pd
import torch.utils.data as Data

def load_data(file_path):
    """
    load data
    :param file_path:
    :return: data[B, N]
    """
    # path
    #data_path = os.path.join("../data/", 'data/', f'{file_path}.csv')
    #data = pd.read_csv(data_path, header=None).values
    
    data_path = os.path.join("./milan/", 'data/', f'{file_path}.npy')
    data = np.load(data_path)
    return data


def data_spilt(data_set, data_ratio, offset, n_his, n_pre, day_slot, c_0=1):
    
    n_route = data_set.shape[1]
    # data size
    n_slot = int(len(data_set) * data_ratio) - n_his - n_pre + 1
    # data scale
    n_scale = n_his + n_pre
    tmp_seq = np.zeros([int(n_slot), n_scale, n_route, c_0])
    for i in range(n_slot):
        # trend
        sta1 = i + int(len(data_set) * offset)
        end1 = sta1 + n_his + n_pre
        df = np.reshape(data_set[sta1:end1, :], [n_his+n_pre, n_route, c_0])
    
        tmp_seq[i, :, :, :] = df

    return tmp_seq

    


def client_data_gen(data_train, data_val, data_test, batch_size):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    data_mean = np.mean(data_train)
    data_std = np.std(data_train)
    # z-score
    data_train = z_score(data_train, data_mean, data_std)
    data_test = z_score(data_test, data_mean, data_std)
    data_val = z_score(data_val, data_mean,data_std)
    # torch
    data_train = TensorFloat(data_train)
    data_val = TensorFloat(data_val)
    data_test = TensorFloat(data_test)
    data_train = Data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False)
    data_val = Data.DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=False)
    data_test = Data.DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=False)
    data = {'train': data_train,
              'val': data_val,
              'test': data_test,
              'mean': data_mean,
              'std': data_std,
           }
       
    return data


def data_gen(file_path, data_assign, n_his, n_pre, day_slot, batch_size):

    # data assign
    init_offset, n_train, n_val, n_test = data_assign
    data_set = load_data(file_path)
    len_df = len(data_set)
    # print(len_df)
    data_set = data_set[-int(len_df*0.2):, :]
    # train
    df_train = data_spilt(data_set, n_train, offset=init_offset, n_his=n_his, n_pre=n_pre, day_slot=day_slot)
    # val
    df_val = data_spilt(data_set, n_val, offset=n_train+init_offset, n_his=n_his, n_pre=n_pre, day_slot=day_slot)
    # test
    df_test = data_spilt(data_set, n_test, offset=n_train+n_val+init_offset, n_his=n_his, n_pre=n_pre, day_slot=day_slot)
    data = client_data_gen(df_train, df_val, df_test, batch_size)

    return data


def data_batch(data, batch_size, shuffle):
    """

    :param data:
    :param batch_size:
    :param shuffle:
    :return: shape [Batch_size, T, N, C_0]
    """
    data_len = len(data)
    data_id = np.arange(data_len)
    # shuffle
    if shuffle:
        np.random.shuffle(data_id)
        # data = data[data_id]

    for st_id in range(0, data_len, batch_size):
        end_id = st_id + batch_size
        if end_id > data_len:
            end_id = data_len

        yield data[data_id[st_id:end_id]]

