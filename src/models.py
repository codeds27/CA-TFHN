import torch
import torch.nn.functional as F
from torch import nn as nn
from torch_geometric.nn import SAGEConv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Context(nn.Module):
    def __init__(self, param_dict):
        super(Context, self).__init__()
        self.org_context_feat_len = param_dict['org_context_feat_len']
        self.enhanced_context_feat_len = param_dict['enhanced_context_feat_len']
        self.context_each_embed = param_dict['context_each_embed']
        self.context_all_len = param_dict['context_all_len']

        self.org_context_embed = nn.Linear(self.org_context_feat_len, self.context_each_embed)
        self.enhanced_context_embed = nn.Linear(self.enhanced_context_feat_len, self.context_each_embed)
        self.context_all_embed = nn.Linear(2 * self.context_each_embed, self.context_all_len)

    def forward(self, sub_graph):
        org_context = sub_graph['org_context']
        org_context = self.org_context_embed(org_context)
        enhanced_context = sub_graph['enhanced_context']
        enhanced_context = self.enhanced_context_embed(enhanced_context)
        # Fuse
        context = torch.cat((org_context, enhanced_context), dim=1)
        context = self.context_all_embed(context)
        return context


class GraphSage(nn.Module):
    def __init__(self, param_dict):
        super(GraphSage, self).__init__()
        self.input_features1 = param_dict['input_features']
        self.hidden_features1 = param_dict['hidden_features']
        self.output_features1 = param_dict['output_features']
        self.conv1 = SAGEConv(in_channels=self.input_features1, out_channels=self.hidden_features1, aggr='mean')
        self.conv2 = SAGEConv(in_channels=self.hidden_features1, out_channels=self.output_features1, aggr='mean')
        self.ac_f1 = nn.ReLU()
        self.ac_f2 = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ac_f1(x)
        x = self.conv2(x, edge_index)
        return self.ac_f2(x)


class MyLSTM(nn.Module):
    def __init__(self, lstm_input_features, lstm_hidden_features, lstm_hidden_num_layers):
        super(MyLSTM, self).__init__()
        self.lstm_input_features = lstm_input_features
        self.lstm_hidden_features = lstm_hidden_features
        self.lstm_hidden_num_layers = lstm_hidden_num_layers
        self.lstm = nn.LSTM(self.lstm_input_features, self.lstm_hidden_features, self.lstm_hidden_num_layers,
                            batch_first=True)
        self.reg = nn.Sequential(
            nn.Linear(self.lstm_hidden_features, self.lstm_hidden_features),
            nn.ReLU()
        )
        self.ac_f1 = nn.ReLU()

    def forward(self, x):
        x, (ht, ct) = self.lstm(x)
        return self.reg(x)


class MySelfAttention(nn.Module):
    def __init__(self, week_count, input_features, num_attention_heads, attention_features):
        super(MySelfAttention, self).__init__()
        self.week_count = week_count
        self.input_features = input_features
        self.num_attention_heads = num_attention_heads
        self.attention_features = attention_features
        self.attention_head_size = int(self.attention_features / self.num_attention_heads)
        self.all_head_size = attention_features

        # Position Embedding
        self.PE = torch.zeros((self.week_count, self.input_features))
        for i in range(1, self.PE.shape[0] + 1):
            for j in range(1, self.PE.shape[1] + 1):
                if j % 2 != 0:
                    twob = j - 1
                    expr = torch.exp(torch.tensor(twob * (-1 * torch.log(torch.tensor(10000 / self.PE.shape[0] + 1)))))
                    self.PE[i - 1][j - 1] = torch.cos(expr * i)
                else:
                    twob = j
                    expr = torch.exp(torch.tensor(twob * (-1 * torch.log(torch.tensor(10000 / self.PE.shape[0] + 1)))))
                    self.PE[i - 1][j - 1] = torch.sin(expr * i)
        self.PE = self.PE.to(device)
        self.key_layer = nn.Linear(self.input_features, self.attention_features)
        self.query_layer = nn.Linear(self.input_features, self.attention_features)
        self.value_layer = nn.Linear(self.input_features, self.attention_features)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_matrix):
        input_matrix = input_matrix + self.PE
        key = self.key_layer(input_matrix)
        query = self.query_layer(input_matrix)
        value = self.value_layer(input_matrix)
        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_probs, value_heads)
        attention_context = attention_context.permute(0, 2, 1, 3).contiguous()
        attention_newsize = attention_context.size()[:-2] + (self.all_head_size,)
        attention_output = attention_context.view(*attention_newsize)

        return attention_output


class Classifier(torch.nn.Module):
    def __init__(self, param_dict):
        super(Classifier, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(param_dict['dnn_input_f1'], param_dict['dnn_hidden_f1']),
            nn.ReLU(),
            nn.Linear(param_dict['dnn_hidden_f1'], param_dict['dnn_hidden_f2']),
            nn.ReLU(),
            nn.Linear(param_dict['dnn_hidden_f2'], param_dict['dnn_hidden_f3']),
            nn.ReLU(),
            nn.Linear(param_dict['dnn_hidden_f3'], param_dict['dnn_output'])
        )

    def forward(self, x):
        return self.dnn(x)


class LGB(nn.Module):
    def __init__(self, param_dict):
        super(LGB, self).__init__()
        self.week_count = param_dict['week_count']
        self.activity_num = param_dict['activity_num']
        self.select_count = param_dict['select_count']

        self.context_embed = Context(param_dict)
        self.gnn = GraphSage(param_dict)
        # TFHN (Time-Flow Hybrid Network)
        self.lstm1 = MyLSTM(param_dict['lstm_input_features'], param_dict['lstm_hidden_features'],
                            param_dict['lstm_hidden_num_layers'])

        self.self_attention1 = MySelfAttention(param_dict['week_count'], param_dict['lstm_hidden_features'],
                                               param_dict['num_attention_heads'],
                                               param_dict['attention_features'])

        self.lstm2 = MyLSTM(param_dict['l2_input_features'], param_dict['l2_hidden_features'],
                            param_dict['l2_hidden_num_layers'])

        self.self_attention2 = MySelfAttention(param_dict['week_count'], param_dict['l2_hidden_features'],
                                               param_dict['s2_num_attention_heads'],
                                               param_dict['s2_attention_features'])

        self.weighted_sum = MySelfAttention(param_dict['week_count'], param_dict['ws_input_features'],
                                            param_dict['ws_num_attention_heads'], param_dict['ws_attention_features'])

        self.classifier = Classifier(param_dict)

    def forward(self, sub_graph):
        batch_size = sub_graph['batch_size']

        context = self.context_embed(sub_graph)
        context = self.gnn(context, sub_graph['edge_index'])
        context_output = context[:batch_size]

        input_matrix = sub_graph['seq_feat'][:batch_size]
        input_matrix = input_matrix.view(batch_size, self.week_count, -1, self.activity_num)
        lstm_input = None
        for i in range(self.week_count):
            x = input_matrix[:, i, :, :]
            act_sum_by_day = torch.sum(x, dim=2).view(batch_size, -1, 1)
            x = torch.cat((x, act_sum_by_day), dim=2)
            act_sum_by_action = torch.sum(x, dim=1).view(batch_size, 1, -1)
            x = torch.cat((x, act_sum_by_action), dim=1)
            x = x.view(batch_size, 1, -1)
            if i == 1:
                lstm_input = x
            else:
                lstm_input = torch.cat((lstm_input, x), dim=1)

        lstm_output = self.lstm1(lstm_input)
        attention_output = self.self_attention1(lstm_output)
        lstm_ouput2 = self.lstm2(attention_output)
        attention_output2 = self.self_attention2(lstm_ouput2)

        context_output = context_output.repeat_interleave(attention_output2.size(1), 0)
        context_output = context_output.view(batch_size, attention_output2.size(1), -1)

        weighted_sum_input = torch.cat((context_output, attention_output2), dim=2)
        all_feat = self.weighted_sum(weighted_sum_input)
        all_mean_feat = torch.mean(all_feat, dim=1)
        pred = self.classifier(all_mean_feat)
        return pred
