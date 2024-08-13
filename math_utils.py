import numpy as np
import tensorflow as tf
import torch
import copy
import pandas as pd

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def l2_penalty(w):
    return (w**2).sum() / 2.0

def nor_adj(W, n):
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)

def gen_adj(file_path, sigma2=0.1, epsilon=0.5):
    #W = pd.read_csv(f'../data/adj/{file_path}.csv', header=None).values
    W = np.load(f'./milan/adj/{file_path}.npy')
    n = W.shape[0]
    W = W / 5000.
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    # refer to Eq.10
    A = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    return nor_adj(A, n)

def gen_01_adj(file_path):
    W = pd.read_csv(f'data/adj/{file_path}.csv', header=None).values
    n = W.shape[0]
    return nor_adj(W, n)

def z_score(data, mean, std):
    return (data - mean) / std


def inverse_score(data, mean, std):
    return data * std + mean


def tensor_mae(true, pre):
    return tf.reduce_mean(tf.abs(true - pre))


def tensor_rmse(true, pre):
    return tf.sqrt(tf.reduce_mean((true - pre) ** 2))


def np_mape(true, pre, mask=True):
    if mask:
        mask_idx = np.where(true > 0.005)
        true = true[mask_idx]
        pre = pre[mask_idx]
    return np.mean(np.abs(np.divide((true - pre), true)))


def np_mae(true, pre):
    return np.mean(np.abs(true - pre))


def np_rmse(true, pre):
    return np.sqrt(np.mean((true - pre) ** 2))


def evaluation(true, pre, mean, std, n_pre):
    true_inverse = inverse_score(true, mean, std)
    pre_inverse = inverse_score(pre, mean, std)

    true_inverse = np.squeeze(true_inverse)
    pre_inverse = np.squeeze(pre_inverse)
    print(true_inverse.shape)
    print(pre_inverse.shape)
    metrics = []
    for i in range(n_pre):
        x_true = true_inverse[:, i, :]
        x_pre = pre_inverse[:, i, :]
        x_mae = np_mae(x_true, x_pre)
        x_rmse = np_rmse(x_true, x_pre)
        x_mape = np_mape(x_true, x_pre)
        metrics.append([x_mae, x_rmse, x_mape])

    return np.array(metrics)

