import numpy as np
import scipy.sparse as sp
import torch


def spatial_shape(adata):
    """Return the 2-D spatial grid shape from ``adata.uns``."""
    return tuple(int(x) for x in adata.uns["binned_data_shape"][:2])


def masked_binned_matrix(adata, mask_key="mask_ecm"):
    """Return flattened binned expression with rows outside mask zeroed."""
    X = adata.uns["binned_data"]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    mask = np.asarray(adata.uns[mask_key]).reshape(-1).astype(np.float32, copy=False)
    X = X.multiply(mask[:, None]).tocsr()
    X.eliminate_zeros()
    return X


def binary_sparse_matrix(X):
    """Return a sparse matrix with the same nonzero pattern and unit values."""
    X_binary = X.copy()
    X_binary.data = np.ones_like(X_binary.data)
    return X_binary


def sparse_sum(X, rows=None):
    """Sum a sparse matrix over columns, optionally after selecting rows."""
    if rows is None:
        summed = X.sum(axis=0)
    else:
        summed = X[rows].sum(axis=0)
    return np.asarray(summed).ravel()


def sparse_mean(X, rows=None):
    """Mean of a sparse matrix over columns, optionally after selecting rows."""
    if rows is None:
        n_rows = X.shape[0]
        summed = sparse_sum(X)
    else:
        n_rows = int(np.asarray(rows).sum()) if np.asarray(rows).dtype == bool else len(rows)
        summed = sparse_sum(X, rows)
    return summed / n_rows if n_rows > 0 else np.full(X.shape[1], np.nan)


def gene_tensor_from_binned(adata, gene_idx, device="cpu", mask_key="mask_ecm"):
    """Extract one binned gene column as a masked 2-D torch tensor."""
    x_shape = spatial_shape(adata)
    gene_map = (
        adata.uns["binned_data"][:, gene_idx]
        .toarray()
        .reshape(x_shape)
        .astype(np.float32, copy=False)
    )
    if mask_key is not None:
        gene_map = gene_map * np.asarray(adata.uns[mask_key], dtype=np.float32)
    return torch.from_numpy(gene_map).float().to(device)


def prep_x_bin_from_binned(adata, pairs, genes=None, device="cpu", mask_key="mask_ecm"):
    """Extract unique gene maps needed by ligand-receptor pairs."""
    genes = list(adata.uns["genes"] if genes is None else genes)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    x_shape = spatial_shape(adata)
    x_bin = {}

    for gp in pairs:
        for gene_name in gp:
            g_idx = gene_to_idx.get(gene_name)
            if g_idx is not None and g_idx not in x_bin:
                x_bin[g_idx] = gene_tensor_from_binned(
                    adata,
                    g_idx,
                    device=device,
                    mask_key=mask_key,
                )

    return x_bin, x_shape, gene_to_idx


def cell_gt_quantile_binary(adata, genes, q=0.75):
    """Return sparse bool matrix for adata[:, genes].X > per-gene quantile."""
    genes = list(genes)
    var_to_idx = {str(gene): i for i, gene in enumerate(adata.var_names)}
    present = [
        (out_idx, var_to_idx[str(gene)])
        for out_idx, gene in enumerate(genes)
        if str(gene) in var_to_idx
    ]

    n_cells = adata.n_obs
    n_genes = len(genes)
    if len(present) == 0:
        return sp.csr_matrix((n_cells, n_genes), dtype=bool)

    out_positions = np.array([out_idx for out_idx, _ in present], dtype=np.int64)
    var_positions = np.array([var_idx for _, var_idx in present], dtype=np.int64)
    X = adata.X[:, var_positions]
    if not sp.issparse(X):
        X_dense = np.asarray(X)
        thresholds = np.quantile(X_dense, q, axis=0)
        present_binary = sp.csr_matrix(X_dense > thresholds)
        rows, cols = present_binary.nonzero()
        mapped_cols = out_positions[cols]
        data = np.ones(rows.shape[0], dtype=bool)
        return sp.csr_matrix((data, (rows, mapped_cols)), shape=(n_cells, n_genes))

    X = X.tocsc()
    rows = []
    cols = []
    interp_pos = (n_cells - 1) * q
    lo = int(np.floor(interp_pos))
    hi = int(np.ceil(interp_pos))
    frac = interp_pos - lo

    for j in range(X.shape[1]):
        start, end = X.indptr[j], X.indptr[j + 1]
        col_rows = X.indices[start:end]
        col_data = X.data[start:end]
        nonzero_data = np.sort(col_data[col_data != 0])
        zero_count = n_cells - nonzero_data.size

        def value_at(pos):
            if pos < zero_count:
                return 0.0
            return nonzero_data[pos - zero_count]

        threshold = (1.0 - frac) * value_at(lo) + frac * value_at(hi)
        keep = col_data > threshold
        if np.any(keep):
            rows.append(col_rows[keep])
            cols.append(np.full(int(keep.sum()), out_positions[j], dtype=np.int64))

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(rows.shape[0], dtype=bool)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        data = np.array([], dtype=bool)

    return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))
