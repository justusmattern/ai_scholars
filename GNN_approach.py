import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from transformers import AutoModel
import pickle
import torch
from torch_geometric.loader import NeighborLoader


with open('hetero_data.pkl', 'rb') as f:
    data = pickle.load(f)

data['paper'].y = data['paper'].y.long()
data['paper'].train_mask = data['paper'].train_mask.long()
data['paper'].test_mask = data['paper'].test_mask.long()
data['author', 'writes', 'paper'].edge_index = data['author', 'writes', 'paper'].edge_index.long()
data['author', 'collaborates', 'author'].edge_index = data['author', 'collaborates', 'author'].edge_index.long()



class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('author', 'collaborates', 'author'): GCNConv(-1, hidden_channels),
                ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):

        paper_text, paper_features = x_dict['paper'][:,list(range(350))], x_dict['paper'][:,list(range(350,362))]

        paper_embeddings = self.scibert(paper_text.long()).pooler_output
        print('paper embeddings', paper_embeddings.shape)

        x_dict['paper'] = torch.cat((paper_embeddings, paper_features), dim=1)

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])

model = HeteroGNN(hidden_channels=64, out_channels=2,
                  num_layers=2)

train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors = {key: [40] * 2 for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=(('paper', data['paper'].train_mask.long()))
)

batch = next(iter(train_loader))
print('batch')
print(batch)

out = model(batch.x_dict, batch.edge_index_dict)
print('out', out)