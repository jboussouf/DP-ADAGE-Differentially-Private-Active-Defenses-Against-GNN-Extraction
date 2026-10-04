"""
GNN Model Architectures
========================
Implements GCN, GAT, GraphSAGE, GIN, and GraphTransformer for use as
victim and surrogate models in the DP-ADAGE evaluation.

All models expose:
  - forward(x, edge_index) → (logits, embedding)
  - get_embedding(x, edge_index) → embedding (penultimate layer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, TransformerConv
    import torch_geometric
except ImportError:
    print("Warning: torch_geometric not found. GNN models will not work without it.")
    
class BaseGNN(nn.Module):
    def get_embedding(self, x, edge_index):
        _, emb = self.forward(x, edge_index)
        return emb

class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = x.clone() # Penultimate layer representation
        x = self.conv2(x, edge_index)
        return x, emb

class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = x.clone()
        x = self.conv2(x, edge_index)
        return x, emb

class GraphSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = x.clone()
        x = self.conv2(x, edge_index)
        return x, emb

class GIN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.conv2 = GINConv(nn2)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = x.clone()
        x = self.conv2(x, edge_index)
        return x, emb

class GraphTransformer(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = x.clone()
        x = self.conv2(x, edge_index)
        return x, emb

def create_model(model_name: str, in_channels: int, hidden_channels: int, out_channels: int, **kwargs) -> nn.Module:
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'GIN': GIN,
        'GraphTransformer': GraphTransformer
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name](in_channels, hidden_channels, out_channels, **kwargs)

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, device='cpu'):
    """Standard training loop for node classification."""
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            
        history['loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
    # Final test
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        history['test_acc'] = test_acc
        
    return history

if __name__ == '__main__':
    print("Testing model instantiations...")
    in_dim, hidden, out_dim = 1433, 64, 7 # Cora-like
    
    models = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'GraphTransformer']
    for m in models:
        model = create_model(m, in_dim, hidden, out_dim)
        print(f"Successfully instantiated {m}")
