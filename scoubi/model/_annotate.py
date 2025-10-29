import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from importlib import resources
from tqdm import trange, tqdm

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # Configure PyTorch for deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

kernel = torch.tensor([[1., 1., 1.],[1., 0., 1.],[1., 1., 1.]]).unsqueeze(0).unsqueeze(0)

class BinAnnotator(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(d, 1, bias=False) for d in input_dims])
        # for fc in self.fcs:
        #     fc.weight.data.fill_(0.2)

    def forward(self, x, temperature=1e-3):
        splits = torch.split(x, [fc.in_features for fc in self.fcs], dim=1)
        out = torch.cat([F.relu(fc(split) / (fc.in_features ** 0.5)) for fc, split in zip(self.fcs, splits)], dim=1)
        return F.softmax(out / temperature, dim=1)

def tstat_loss(X, P, eps=1e-8):
    p1 = P.unsqueeze(1)
    p2 = 1 - P.unsqueeze(1)
    n1 = p1.sum()
    n2 = p2.sum()
    mu1 = torch.sum(p1 * X, dim=0) / (n1 + eps)
    mu2 = torch.sum(p2 * X, dim=0) / (n2 + eps)
    var1 = torch.sum(p1 * (X - mu1)**2, dim=0) / (n1 + eps)
    var2 = torch.sum(p2 * (X - mu2)**2, dim=0) / (n2 + eps)
    t_stat = (mu1 - mu2) / torch.sqrt(var1 / n1 + var2 / n2 + eps)
    return -t_stat.mean()

def do_conv(matrix, kernel=None):
    return F.conv2d(matrix[None, None, :, :], kernel, padding='same')[0, 0]

def get_zscore(Z_1, Z_2, x_a, x_b, kernel, eps = 1e-30):
    agg_Z_2 = do_conv(Z_2, kernel)
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel)
    Z_prod = Z_1 * agg_Z_2

    N = torch.sum(Z_prod)
    X = torch.sum((Z_1 * x_a) * agg_Z_2_x_b)
    p_a = torch.sum(x_a * Z_1) / (torch.sum(Z_1) + eps)
    p_d = torch.sum(x_b * Z_2) / (torch.sum(Z_2) + eps)
    E_x = N * p_a * p_d
    E_x_2 = torch.sum((Z_prod * p_a * p_d)**2) + (torch.sum(Z_prod)**2 - torch.sum(Z_prod**2)) * (p_a * p_d)**2
    var_X = E_x + E_x_2 - E_x**2
    var_X = torch.clamp(var_X, min=eps)
    z_score = (X - E_x) / torch.sqrt(var_X)
    return z_score

def _prep_dict(array_usr, pairs, genes, device):
    x_bin = {}
    x_shape = (array_usr.shape[0], array_usr.shape[1])
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    for gp in pairs:
        for gene_name in gp:
            g_idx = gene_to_idx.get(gene_name)
            if g_idx is not None and g_idx not in x_bin:
                x_bin[g_idx] = torch.from_numpy(array_usr[:, :, g_idx]).float().to(device)
    return x_bin, x_shape, gene_to_idx

def train(adata, axon_markers, dendrite_markers, pairs = None, epochs = 500, lambda_gex = 1e-1, ftemp = 1e-1, patience = 50, lr = 1e-3, device='cpu', weight_decay=1e-4, seed = 42):
    set_seed(seed)
    if pairs is None:
        with resources.open_binary("scoubi.data", "pairs.pkl") as fp:
            pairs = pickle.load(fp)
    genes = list(adata.uns['genes'])
    pairs = [pair for pair in pairs if pair[0] in genes and pair[1] in genes]
    array_usr = (adata.uns['binned_data'] * adata.uns["mask_ecm"][:, :, None]).copy()
    axon_markers = [gene for gene in axon_markers if gene in genes]
    dendrite_markers = [gene for gene in dendrite_markers if gene in genes]
    axon_idx = [genes.index(gene) for gene in axon_markers]
    dendrite_idx = [genes.index(gene) for gene in dendrite_markers]

    taxon_markers = {"Axon": axon_markers, "Dendrite": dendrite_markers}
    num_cts = len(taxon_markers)
    selected_cts = list(taxon_markers.keys())
    ct_genes, len_idx, ct_genes_idx = [], [], []
    for ct in selected_cts:
        ct_genes.extend(taxon_markers[ct])
        len_idx.append(len(taxon_markers[ct]))
        for gene in taxon_markers[ct]:
            ct_genes_idx.append(genes.index(gene))
    
    x_all = torch.from_numpy(array_usr[:, :, ct_genes_idx]).float().flatten(0, 1).to(device)
    x_mask = (x_all.sum(axis = 1) != 0).unsqueeze(1).float().clone()
    ct_markers_gex = {ct: torch.from_numpy(array_usr[:, :, [genes.index(x) for x in taxon_markers[ct]]]).float().flatten(0, 1).to(device) for ct in selected_cts}

    x_bin, x_shape, gene_to_idx = _prep_dict(array_usr, pairs, genes, device)
    
    model = BinAnnotator(input_dims=len_idx).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    patience_counter = 0
    best_loss = float('inf')
    best_model_weights = None

    pbar = trange(epochs, desc="Training", unit="epoch")
    for epoch in pbar:
        optimizer.zero_grad()
        temperature = 1e-3
        Z = model(x_all, temperature)
        Z = Z * x_mask

        # GEX loss
        gex_loss = 0.0
        for n, ct_name in enumerate(selected_cts):
            gex_loss += tstat_loss(ct_markers_gex[ct_name], Z[:, n])
        gex_loss_term = (lambda_gex * gex_loss) / num_cts

        # Communication loss
        Z_a = Z[:, 0].reshape(x_shape)
        Z_d = Z[:, 1].reshape(x_shape)
        ct_loss_for_n1 = 0.0
        for gp in pairs:
            a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
            if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin: 
                continue
            x_a, x_b = x_bin[a_idx], x_bin[b_idx]
            ct_loss_for_n1 -= get_zscore(Z_a, Z_d, x_a, x_b, kernel.to(device))

        total_loss = gex_loss_term + (ct_loss_for_n1 / len(pairs))
        total_loss.backward()
        optimizer.step()

        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} | Total: {total_loss.item():.6f} | GEX: {gex_loss_term.item():.6f} | Comm: {(ct_loss_for_n1 / len(pairs)).item():.6f} | Patience: {patience_counter}/{patience}"
        )

        # Early stopping logic
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Loss did not improve for {patience} epochs.")
                break

    if best_model_weights:
        model.load_state_dict(best_model_weights)
    adata.uns["model_weights"] = {k: v.cpu().numpy() for k, v in best_model_weights.items()}
    model.eval()
    with torch.no_grad():
        Z = model(x_all, temperature=ftemp).cpu().numpy()
    adata.uns["bin_probabilities"] = Z
    adata.uns['lr_pairs'] = pairs
    adata.uns['axon_markers'] = axon_markers
    adata.uns['dendrite_markers'] = dendrite_markers
    return adata 