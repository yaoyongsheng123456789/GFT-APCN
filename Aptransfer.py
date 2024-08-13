import numpy as np
import tensorflow.compat.v1 as tf
import torch.nn.functional as F
tf.disable_v2_behavior()
import torch
import copy
import pandas as pd

def Aptransfer(w,d):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):# i: 参与训练的 clients_num
            w_avg[k] += d[i]*w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg