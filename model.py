from GCN import RGCN_cell

import torch
import torch.nn.functional as F
import torch.nn as nn


class RGCN(nn.Module):
    def __init__(self, args):
        super(RGCN, self).__init__()
        self.input_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.out_dim
        self.num_layers = args.num_layers
        self.n_pre = args.n_pre
        self.n_his = args.n_his
        self.embedding_dim = args.embedding_dim
        self.rgcn_cells = nn.ModuleList()
        self.rgcn_cells.append(RGCN_cell(input_dim=self.input_dim, out_dim=self.hid_dim))
        for i in range(1, self.num_layers):
            self.rgcn_cells.append(RGCN_cell(input_dim=self.hid_dim, out_dim=self.hid_dim))
        
        self.output_cells = RGCN_cell(input_dim=self.hid_dim, out_dim=self.hid_dim)
       # pam
        self.time_en = nn.Parameter(torch.FloatTensor(self.n_his, self.n_pre))
        self.time_en_w = nn.Parameter(torch.FloatTensor(self.hid_dim, self.hid_dim))
        self.time_en_b = nn.Parameter(torch.FloatTensor(self.hid_dim))


        # out
        self.time_conv = nn.Parameter(torch.FloatTensor(self.hid_dim, 64))
        self.time_conv_bias = nn.Parameter(torch.FloatTensor(64))
        self.time_out = nn.Parameter(torch.FloatTensor(64, self.out_dim))
        self.time_out_bias = nn.Parameter(torch.FloatTensor(self.out_dim))

    def time_att(self, inputs):
        _, T, N, C = inputs.shape
        W = self.time_en
        output = torch.einsum('btnc,to->bonc', inputs, W)
        output = torch.einsum('btnc,co->btno', output, self.time_en_w) + self.time_en_b
#         output = torch.transpose(self.time_conv(inputs_mul), 1, 3)

        return torch.tanh(output)
# normal
#     def forward(self, inputs, adj, node_num):
#         current_inputs = inputs
#         for layer_id in range(self.num_layers):
#             state = self.rgcn_cells[layer_id].init_hidden_state(inputs.shape[0], node_num)
#             cell = self.rgcn_cells[layer_id]
#             inner_state = []
#             for t in range(self.n_his):
#                 state = cell(inputs=current_inputs[:, t, :, :], state=state, adj=adj, act = torch.tanh)
#                 inner_state.append(state)

#             current_inputs = torch.stack(inner_state, dim=1)
          
#         output_trans = self.time_att(current_inputs)
        
#         state = current_inputs[:, 0, :, :]
#         inner_state = []
#         for t in range(self.n_pre):
#             state = self.output_cells(inputs=output_trans[:, t, :, :], state=state, adj=adj, act = torch.tanh)
#             inner_state.append(state)

#         current_outputs = torch.stack(inner_state, dim=1)
        
#         output = torch.einsum('btnc,co->btno', current_outputs, self.time_conv) + self.time_conv_bias
#         output = torch.einsum('btnc,co->btno', output, self.time_out) + self.time_out_bias
#         return output
        
    def forward(self, inputs, adj, node_num):
        current_inputs = inputs
        for layer_id in range(self.num_layers):
            state = self.rgcn_cells[layer_id].init_hidden_state(inputs.shape[0], node_num)
            cell = self.rgcn_cells[layer_id]
            inner_state = []
            for t in range(self.n_his):
                state = cell(inputs=current_inputs[:, t, :, :], state=state, adj=adj, act = torch.tanh)
                inner_state.append(state)

            current_inputs = torch.stack(inner_state, dim=1)
          
        output_trans = self.time_att(current_inputs)
        
        state = current_inputs[:, -1, :, :]
        # print(state.shape)
        inner_state = []
        for t in range(self.n_pre):
            state = self.output_cells(inputs=output_trans[:, t, :, :], state=state, adj=adj, act = torch.tanh)
            inner_state.append(state)

        current_outputs = torch.stack(inner_state, dim=1)
        
        output = torch.einsum('btnc,co->btno', current_outputs, self.time_conv) + self.time_conv_bias
        output = torch.einsum('btnc,co->btno', output, self.time_out) + self.time_out_bias
        
        
        return output