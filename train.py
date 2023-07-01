import argparse
import os
import pickle
import torch
import numpy as np

from torch import optim
from src.models import *
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

myparser = argparse.ArgumentParser(description='optional parameters')
myparser.add_argument('-indir', type=str, help='input dir (default: current dir)', default='.')
myparser.add_argument('-outdir', type=str, help='output dir (default: current dir)', default='.')
myparser.add_argument('-e', type=int, default=15, help='epoch (default: 15)')
myparser.add_argument('-r', type=int, default=0, help='random seed (default: 0)')
myparser.add_argument('-lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

args = myparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dir = os.path.abspath(os.path.expanduser(str(args.indir)))
output_dir = os.path.abspath(os.path.expanduser(str(args.outdir)))
epoch_num = args.e
learning_rate = args.lr

# load the graph
with open(os.path.join(input_dir, 'StrongClassmatesGraph.pkl'), 'rb') as f:
    graph = pickle.load(f)
graph = graph.to(device)

# create graph loader
train_loader = NeighborLoader(
    data=graph,
    num_neighbors=[8, 4],
    input_nodes=graph.train_mask,
    batch_size=256,
    shuffle=True
)

test_loader = NeighborLoader(
    data=graph,
    num_neighbors=[8, 4],
    input_nodes=graph.test_mask,
    batch_size=128,
    shuffle=True
)

# set the hyper parameters
param_dict = dict({
    # Constant
    'activity_num': 22, 'sta_day': 35, 'week_count': 5, 'select_count': 5,
    # Context-Embedding
    'org_context_feat_len': 7, 'enhanced_context_feat_len': 32, 'context_each_embed': 16, 'context_all_len': 16,
    # GraphSage
    'input_features': 16, 'hidden_features': 32, 'output_features': 16,
    # LSTM of first block in TFHN
    'lstm_input_features': 184, 'lstm_hidden_features': 128, 'lstm_hidden_num_layers': 1,
    # Self-Attention of first block in TFHN
    'num_attention_heads': 1, 'attention_features': 64,
    # LSTM2 of second block in TFHN
    'l2_input_features': 64, 'l2_hidden_features': 32, 'l2_hidden_num_layers': 1,
    # Self-Attention of second block in TFHN
    's2_num_attention_heads': 1, 's2_attention_features': 16,
    # Weighted-Sum
    'ws_num_attention_heads': 1, 'ws_input_features': 32, 'ws_attention_features': 16,
    # DNN
    'dnn_input_f1': 16, 'dnn_hidden_f1': 16, 'dnn_hidden_f2': 8, 'dnn_hidden_f3': 4, 'dnn_output': 1
})
# create the model
model = LGB(param_dict)
model = model.to(device)

# select the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(epoch_num):
    total_loss, total_examples = 0, 0
    for sub_graph in tqdm(train_loader):
        optimizer.zero_grad()
        pred = model(sub_graph)
        batch_size = sub_graph['batch_size']
        ground_truth = sub_graph['labels'][:batch_size].view(-1, 1).to(torch.float)
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    preds = []
    ground_truths = []
    for sub_graph in tqdm(test_loader):
        with torch.no_grad():
            batch_size = sub_graph['batch_size']
            pred = model(sub_graph)
            preds.append(pred)
            truth = sub_graph['labels'][:batch_size].view(-1, 1)
            ground_truths.append(truth)
    pred = torch.cat(preds, dim=0).cpu()
    pred_label = torch.sigmoid(pred).numpy()
    pred_label[pred_label < 0.5] = 0
    pred_label[pred_label >= 0.5] = 1
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    acc = np.sum(pred_label == ground_truth) / len(ground_truth)
    precision = precision_score(ground_truth, pred_label)
    recall = recall_score(ground_truth, pred_label)
    auc = roc_auc_score(ground_truth, pred)
    f1 = f1_score(ground_truth, pred_label)

    print(f"Epoch: {epoch:03d}, Test ACC: {acc:.4f}")
    print(f"Epoch: {epoch:03d}, Test Precision: {precision:.4f}")
    print(f"Epoch: {epoch:03d}, Test Recall: {recall:.4f}")
    print(f"Epoch: {epoch:03d}, Test AUC: {auc:.4f}")
    print(f"Epoch: {epoch:03d}, Test F1: {f1:.4f}")
