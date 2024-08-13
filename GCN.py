import torch
import torch.nn.functional as F
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weights = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(self.out_dim))

    def forward(self, inputs, adj):
        # in B, N, C
        # out B, N, C_out
        N, C = inputs.shape[1], inputs.shape[-1]
        # B, N, C -> B, C, N -> B*C, N
        inputs_res = torch.reshape(torch.transpose(inputs, 1, 2), [-1, N])
        # B, N, C
        inputs_mul = torch.transpose(torch.reshape(torch.matmul(inputs_res, adj), [-1, self.in_dim, N]), 1, 2)
        # B, N, COUT
        output = torch.reshape(torch.matmul(torch.reshape(inputs_mul, [-1, self.in_dim]), self.weights), [-1, N, self.out_dim]) + self.bias

        return output


class RGCN_cell(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(RGCN_cell, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.GCN_gate = GCN(in_dim=self.input_dim+self.out_dim, out_dim=self.out_dim*2)
        self.GCN_update = GCN(in_dim=self.input_dim+self.out_dim, out_dim=self.out_dim)

    def forward(self, inputs, state, adj, act):
        state = state.to(inputs.device)
        inputs_state = torch.cat((inputs, state), dim=-1)
        z_r = torch.sigmoid(self.GCN_gate(inputs=inputs_state, adj=adj))
        z, r = torch.split(z_r, split_size_or_sections=self.out_dim, dim=-1)
        inputs_candidate = torch.cat((r * state, inputs), dim=-1)
        h_hat = act(self.GCN_update(inputs=inputs_candidate, adj=adj))
        h = (1 - z) * state + z * h_hat
        return h

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.out_dim)