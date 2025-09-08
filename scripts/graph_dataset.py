import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pandas as pd


class GraphDataset:
    """Minimal loader for symbol graphs stored as JSON."""

    def __init__(self, graph_file: str | Path):
        path = Path(graph_file)
        with open(path) as f:
            data = json.load(f)
        self.symbols: List[str] = data.get("symbols", [])
        edge_index = torch.tensor(data.get("edge_index", []), dtype=torch.long)
        if edge_index.numel():
            edge_index = edge_index.t().contiguous()
        self.data = Data(edge_index=edge_index, num_nodes=len(self.symbols))
        weights = data.get("edge_weight")
        if weights is not None:
            self.data.edge_weight = torch.tensor(weights, dtype=torch.float)


class SymbolGNN(torch.nn.Module):
    """Simple two-layer GraphSAGE network producing node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def compute_gnn_embeddings(
    df: pd.DataFrame,
    dataset: GraphDataset,
    state_dict: Optional[Dict[str, List[List[float]]]] = None,
    epochs: int = 50,
) -> Tuple[Dict[str, List[float]], Dict[str, List[List[float]]]]:
    """Return node embeddings and optionally update ``state_dict``.

    If ``state_dict`` is provided it is loaded instead of training.
    """

    if "symbol" not in df.columns:
        return {}, {}
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("graph_emb")
    ]
    sym_feats = df.groupby("symbol")[numeric_cols].mean()
    sym_feats = sym_feats.reindex(dataset.symbols).fillna(0.0)
    x = torch.tensor(sym_feats.to_numpy(dtype=float), dtype=torch.float)

    if state_dict:
        w1 = torch.tensor(state_dict["conv1.lin_l.weight"])  # type: ignore[index]
        hidden_dim = w1.shape[0]
        w2 = torch.tensor(state_dict["conv2.lin_l.weight"])  # type: ignore[index]
        out_dim = w2.shape[0]
        model = SymbolGNN(x.size(1), hidden_dim, out_dim)
        model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})
    else:
        hidden_dim = min(16, x.size(1))
        out_dim = min(8, hidden_dim)
        model = SymbolGNN(x.size(1), hidden_dim, out_dim)
        if x.numel() > 0 and dataset.data.edge_index.numel() > 0:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(max(1, epochs)):
                opt.zero_grad()
                out = model(x, dataset.data.edge_index)
                loss = F.mse_loss(out, x[:, : out.size(1)])
                loss.backward()
                opt.step()
    model.eval()
    with torch.no_grad():
        emb = model(x, dataset.data.edge_index).cpu().numpy()
    embeddings = {sym: emb[i].tolist() for i, sym in enumerate(dataset.symbols)}
    state = {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}
    return embeddings, state
