import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from importlib import resources
from tqdm import trange

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
    """Linear marker model that scores bins for each neurite compartment."""

    def __init__(self, input_dims):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(d, 1, bias=False) for d in input_dims])
        # for fc in self.fcs:
        #     fc.weight.data.fill_(0.2)

    def forward(self, x, temperature=1e-3):
        splits = torch.split(x, [fc.in_features for fc in self.fcs], dim=1)
        out = torch.cat([F.relu(fc(split) / (fc.in_features ** 0.5)) for fc, split in zip(self.fcs, splits)], dim=1)
        return F.softmax(out / temperature, dim=1)

    def score(self, x):
        splits = torch.split(x, [fc.in_features for fc in self.fcs], dim=1)
        out = torch.cat([F.relu(fc(split) / (fc.in_features ** 0.5)) for fc, split in zip(self.fcs, splits)], dim=1)
        return out
    
def tstat_loss(X, P, eps=1e-8):
    """Return the negative two-group t-statistic objective for marker expression."""
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
    """Apply a same-size 2-D convolution to one spatial map."""
    return F.conv2d(matrix[None, None, :, :], kernel, padding='same')[0, 0]

def precompute_zscore_shared(Z_1, Z_2, kernel, eps=1e-30):
    """Compute epoch-level invariants shared across all pairs."""
    agg_Z_2 = do_conv(Z_2, kernel)
    Z_prod = Z_1 * agg_Z_2
    N = torch.sum(Z_prod)
    Z_prod_sq_sum = torch.sum(Z_prod ** 2)
    Z_1_sum = torch.sum(Z_1)
    Z_2_sum = torch.sum(Z_2)
    return agg_Z_2, Z_prod, N, Z_prod_sq_sum, Z_1_sum, Z_2_sum


def get_zscore_pair(Z_1, Z_2, x_a, x_b, kernel,
                    agg_Z_2, Z_prod, N, Z_prod_sq_sum,
                    Z_1_sum, Z_2_sum, eps=1e-30, iid=True):
    """Compute z-score for a single pair using pre-computed shared tensors."""
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel)
    X = torch.sum((Z_1 * x_a) * agg_Z_2_x_b)
    p_a = torch.sum(x_a * Z_1) / (Z_1_sum + eps)
    p_d = torch.sum(x_b * Z_2) / (Z_2_sum + eps)
    E_x = N * p_a * p_d
    if iid:
        var_X = torch.clamp(E_x * (1 - p_a * p_d), min=eps)
    else:
        E_x_2 = Z_prod_sq_sum * (p_a * p_d) ** 2 + (N ** 2 - Z_prod_sq_sum) * (p_a * p_d) ** 2
        var_X = torch.clamp(E_x + E_x_2 - E_x ** 2, min=eps)
    return (X - E_x) / torch.sqrt(var_X)


def get_zscore(Z_1, Z_2, x_a, x_b, kernel, eps=1e-30, iid=True):
    """Compute one ligand-receptor communication z-score."""
    shared = precompute_zscore_shared(Z_1, Z_2, kernel, eps)
    return get_zscore_pair(Z_1, Z_2, x_a, x_b, kernel, *shared, eps=eps, iid=iid)


def get_zscore_pairs_batched(
    Z_1,
    Z_2,
    x_bin,
    pairs,
    gene_to_idx,
    kernel,
    shared=None,
    eps=1e-30,
    iid=True,
    receptor_batch_size=32,
):
    """
    Compute z-scores for many LR pairs while preserving get_zscore_pair behavior.

    This avoids repeating the expensive ``conv(Z_2 * x_b)`` operation for every
    pair by batching unique receptor genes. Pair order and skipped-pair behavior
    match the current training loop: only pairs with both genes present in
    ``x_bin`` are returned.
    """
    if shared is None:
        shared = precompute_zscore_shared(Z_1, Z_2, kernel, eps)
    _, Z_prod, N, Z_prod_sq_sum, Z_1_sum, Z_2_sum = shared
    z_device = Z_1.device
    z_dtype = Z_1.dtype
    expr_cache = {}

    def _expr(g_idx):
        expr = expr_cache.get(g_idx)
        if expr is None:
            expr = x_bin[g_idx].to(device=z_device, dtype=z_dtype, non_blocking=True)
            expr_cache[g_idx] = expr
        return expr

    valid_pairs = []
    receptor_indices = []
    for gp in pairs:
        a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
        if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin:
            continue
        valid_pairs.append((gp, a_idx, b_idx))
        if b_idx not in receptor_indices:
            receptor_indices.append(b_idx)

    if not valid_pairs:
        return torch.empty(0, dtype=Z_1.dtype, device=Z_1.device), []

    ligand_prob = {}
    receptor_prob = {}
    for _, a_idx, b_idx in valid_pairs:
        if a_idx not in ligand_prob:
            ligand_prob[a_idx] = torch.sum(_expr(a_idx) * Z_1) / (Z_1_sum + eps)
        if b_idx not in receptor_prob:
            receptor_prob[b_idx] = torch.sum(_expr(b_idx) * Z_2) / (Z_2_sum + eps)

    pairs_by_receptor = {b_idx: [] for b_idx in receptor_indices}
    for pair_pos, (_, a_idx, b_idx) in enumerate(valid_pairs):
        pairs_by_receptor[b_idx].append((pair_pos, a_idx))

    zscores = [None] * len(valid_pairs)
    receptor_batch_size = max(1, int(receptor_batch_size))
    for start in range(0, len(receptor_indices), receptor_batch_size):
        receptor_chunk = receptor_indices[start:start + receptor_batch_size]
        receptor_stack = torch.stack([Z_2 * _expr(b_idx) for b_idx in receptor_chunk])
        agg_chunk = F.conv2d(receptor_stack[:, None, :, :], kernel, padding='same')[:, 0, :, :]

        for chunk_pos, b_idx in enumerate(receptor_chunk):
            agg_Z_2_x_b = agg_chunk[chunk_pos]
            p_d = receptor_prob[b_idx]
            for pair_pos, a_idx in pairs_by_receptor[b_idx]:
                p_a = ligand_prob[a_idx]
                X = torch.sum((Z_1 * _expr(a_idx)) * agg_Z_2_x_b)
                E_x = N * p_a * p_d
                if iid:
                    var_X = torch.clamp(E_x * (1 - p_a * p_d), min=eps)
                else:
                    E_x_2 = Z_prod_sq_sum * (p_a * p_d) ** 2 + (N ** 2 - Z_prod_sq_sum) * (p_a * p_d) ** 2
                    var_X = torch.clamp(E_x + E_x_2 - E_x ** 2, min=eps)
                zscores[pair_pos] = (X - E_x) / torch.sqrt(var_X)

    return torch.stack(zscores), [gp for gp, _, _ in valid_pairs]


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


def _gene_tensor_from_binned(binned_data, x_shape, gene_idx, mask_ecm):
    gene_map = binned_data[:, gene_idx].toarray().reshape(x_shape).astype(np.float32, copy=False)
    gene_map = gene_map * mask_ecm
    return torch.from_numpy(gene_map).float()


def _clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _prep_training_tensors(adata, pairs, genes, ct_genes_idx, device):
    x_shape = tuple(adata.uns['binned_data_shape'][:2])
    binned_data = adata.uns['binned_data'].tocsc()
    mask_ecm = adata.uns['mask_ecm'].astype(np.float32, copy=False)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    required_idx = set(ct_genes_idx)
    for gp in pairs:
        for gene_name in gp:
            g_idx = gene_to_idx.get(gene_name)
            if g_idx is not None:
                required_idx.add(g_idx)

    tensor_cache = {
        g_idx: _gene_tensor_from_binned(binned_data, x_shape, g_idx, mask_ecm).to(
            device=device,
            non_blocking=True,
        )
        for g_idx in sorted(required_idx)
    }
    x_bin = {
        g_idx: tensor_cache[g_idx]
        for gp in pairs
        for gene_name in gp
        for g_idx in [gene_to_idx.get(gene_name)]
        if g_idx is not None and g_idx in tensor_cache
    }
    return tensor_cache, x_bin, x_shape, gene_to_idx


def train(adata, axon_markers, dendrite_markers, pairs = None, epochs = 500, lambda_gex = 1e-1, lambda_comm = 1, ftemp = 1e-1, patience = 50, lr = 1e-3, device='cpu', weight_decay=1e-4, seed = 42, tol = 1e-2, do_comm = True, iid=True, zscore_backend="batched", zscore_batch_size=32):
    """Train the BinAnnotator model to assign spatial bins to axon or dendrite identity.

    Optimizes a joint loss combining a GEX alignment term (t-statistic between
    annotated compartments) and a communication term (z-score enrichment of
    ligand-receptor co-localisation across neighbouring bins).

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics object. Must contain ``adata.uns['binned_data']``,
        ``adata.uns['binned_data_shape']``, ``adata.uns['mask_ecm']``, and
        ``adata.uns['genes']``. Training extracts only marker and ligand-receptor
        gene maps from the sparse binned matrix; it does not materialize the full
        dense ``(x_bins, y_bins, n_genes)`` cube.
    axon_markers : list of str
        Gene names used as axon compartment markers.
    dendrite_markers : list of str
        Gene names used as dendrite compartment markers.
    pairs : list of (str, str) or None, optional
        Ligand-receptor gene pairs for the communication loss. Defaults to the
        bundled ``pairs.pkl`` resource.
    epochs : int, optional
        Maximum number of training epochs (default 500).
    lambda_gex : float, optional
        Weight for the GEX alignment loss term (default 0.1).
    lambda_comm : float, optional
        Weight for the communication loss term (default 1).
    ftemp : float, optional
        Softmax temperature used when computing final bin probabilities after
        training (default 0.1).
    patience : int, optional
        Early-stopping patience in epochs (default 50).
    lr : float, optional
        AdamW learning rate (default 1e-3).
    device : str, optional
        Torch device string, e.g. ``'cpu'`` or ``'cuda'``. Selected training
        tensors and the model are moved to this device. Default: ``'cpu'``.
    weight_decay : float, optional
        AdamW weight decay (default 1e-4).
    seed : int, optional
        Random seed for reproducibility (default 42).
    tol : float, optional
        Minimum loss improvement required to reset the patience counter
        (default 1e-2).
    do_comm : bool, optional
        Whether to include the communication loss (default True).
    iid : bool, optional
        Variance model for the communication z-score. ``True`` (default) uses
        the fast closed-form ``var = E[X] * (1 - p_a * p_d)``; ``False`` uses
        the full spatial covariance formula.
    zscore_backend : {"loop", "batched"}, optional
        Communication z-score execution backend. ``"batched"`` uses the same
        formula as the original path but batches repeated receptor-gene
        convolutions. Use ``"loop"`` to run the original per-pair convolution
        path. Default: ``"batched"``.
    zscore_batch_size : int, optional
        Number of unique receptor genes to convolve at once when
        ``zscore_backend="batched"``. Default: 32.
    Returns
    -------
    AnnData
        The input ``adata`` updated in-place with the following keys in
        ``adata.uns``: ``'model_weights'``, ``'bin_probabilities'``,
        ``'bin_scores'``, ``'lr_pairs'``, ``'axon_markers'``,
        ``'dendrite_markers'``.
    """
    set_seed(seed)
    if pairs is None:
        with resources.open_binary("scoubi.data", "pairs.pkl") as fp:
            pairs = pickle.load(fp)
    genes = list(adata.uns['genes'])
    pairs = [pair for pair in pairs if pair[0] in genes and pair[1] in genes]
    axon_markers = [gene for gene in axon_markers if gene in genes]
    dendrite_markers = [gene for gene in dendrite_markers if gene in genes]
    print(f"Using {len(axon_markers)} axon markers: {axon_markers}")
    print(f"Using {len(dendrite_markers)} dendrite markers: {dendrite_markers}")
    print(f"Using {len(pairs)} ligand-receptor pairs.")

    taxon_markers = {"Axon": axon_markers, "Dendrite": dendrite_markers}
    num_cts = len(taxon_markers)
    selected_cts = list(taxon_markers.keys())
    ct_genes, len_idx, ct_genes_idx = [], [], []
    for ct in selected_cts:
        ct_genes.extend(taxon_markers[ct])
        len_idx.append(len(taxon_markers[ct]))
        for gene in taxon_markers[ct]:
            ct_genes_idx.append(genes.index(gene))

    tensor_cache, x_bin, x_shape, gene_to_idx = _prep_training_tensors(
        adata, pairs, genes, ct_genes_idx, device
    )

    x_all = torch.stack(
        [tensor_cache[g_idx].to(device=device, non_blocking=True) for g_idx in ct_genes_idx],
        dim=-1,
    ).flatten(0, 1)
    x_mask = (x_all.sum(axis = 1) != 0).unsqueeze(1).float().clone()
    ct_markers_gex = {}
    start = 0
    for ct, width in zip(selected_cts, len_idx):
        ct_markers_gex[ct] = x_all[:, start:start + width]
        start += width
    
    model = BinAnnotator(input_dims=len_idx).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    patience_counter = 0
    best_loss = float('inf')
    best_model_weights = None

    kernel_dev = kernel.to(device)
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
        if do_comm:
            shared = precompute_zscore_shared(Z_a, Z_d, kernel_dev)
            if zscore_backend == "loop":
                epoch_expr_cache = {}
                def _epoch_expr(g_idx):
                    expr = epoch_expr_cache.get(g_idx)
                    if expr is None:
                        expr = x_bin[g_idx].to(device=Z_a.device, dtype=Z_a.dtype, non_blocking=True)
                        epoch_expr_cache[g_idx] = expr
                    return expr

                for gp in pairs:
                    a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
                    if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin:
                        continue
                    x_a = _epoch_expr(a_idx)
                    x_b = _epoch_expr(b_idx)
                    ct_loss_for_n1 -= get_zscore_pair(Z_a, Z_d, x_a, x_b, kernel_dev, *shared, iid=iid)
            elif zscore_backend == "batched":
                zscores, _ = get_zscore_pairs_batched(
                    Z_a,
                    Z_d,
                    x_bin,
                    pairs,
                    gene_to_idx,
                    kernel_dev,
                    shared=shared,
                    iid=iid,
                    receptor_batch_size=zscore_batch_size,
                )
                ct_loss_for_n1 = -torch.sum(zscores) if zscores.numel() else torch.zeros((), dtype=Z_a.dtype, device=Z_a.device)
            else:
                raise ValueError("zscore_backend must be 'loop' or 'batched'")

        comm_denominator = len(pairs) if len(pairs) else 1
        total_loss = gex_loss_term + (lambda_comm * ct_loss_for_n1 / comm_denominator)
        total_loss.backward()
        optimizer.step()

        comm_term = (ct_loss_for_n1 / comm_denominator).item() if do_comm else ct_loss_for_n1
        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} | Total: {total_loss.item():.6f} | GEX: {gex_loss_term.item():.6f} | Comm: {comm_term:.6f} | Patience: {patience_counter}/{patience}"
        )

        # Early stopping logic
        current_loss = total_loss.item()
        if current_loss < best_loss - tol:
            best_loss = current_loss
            patience_counter = 0
            best_model_weights = _clone_state_dict(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Loss did not improve for {patience} epochs.")
                break

    if best_model_weights is None:
        best_model_weights = _clone_state_dict(model.state_dict())
    else:
        model.load_state_dict(best_model_weights)
    adata.uns["model_weights"] = {k: v.cpu().numpy() for k, v in best_model_weights.items()}
    model.eval()
    with torch.no_grad():
        Z = model(x_all, temperature=ftemp).cpu().numpy()
        Z_score = model.score(x_all).cpu().numpy()
    adata.uns["bin_probabilities"] = Z
    adata.uns['lr_pairs'] = pairs
    adata.uns['axon_markers'] = axon_markers
    adata.uns['dendrite_markers'] = dendrite_markers
    adata.uns['bin_scores'] = Z_score
    return adata 
