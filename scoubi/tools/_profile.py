import numpy as np
import pandas as pd
import torch
from ..model import do_conv, _prep_dict, kernel
from scipy.spatial import cKDTree

def get_whisper_edges(Z_1, Z_2, x_a, x_b, device):
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel=kernel.to(device))
    X = (Z_1 * x_a) * agg_Z_2_x_b
    return X

def expression_profile(adata, key="cell_type", threshold=None, normalize=False):
    """
    Build a gene-expression profile of interface bins grouped by a cell-level annotation.

    For each interface bin (from ``adata.uns['interface_map']``), the nearest cell is
    looked up via the pre-computed KNN index (``adata.uns['interface_knn_idx']``), and
    its annotation label is used to group bins.  The per-group mean binary (or
    normalised) expression across genes is returned as a DataFrame and stored in
    ``adata.uns``.

    Parameters
    ----------
    adata : ad.AnnData
        Must contain ``adata.obs[key]``, ``adata.uns['interface_map']``,
        ``adata.uns['interface_knn_idx']``, ``adata.uns['binned_data']``,
        ``adata.uns['binned_data_shape']``, ``adata.uns['mask_ecm']``,
        ``adata.uns['genes']``, ``adata.uns['axon_markers']``, and
        ``adata.uns['dendrite_markers']``.
    key : str, optional
        Column in ``adata.obs`` used to group interface bins.
        Default: ``'cell_type'``.
    threshold : float, optional
        Unused (reserved for future filtering).  Default: ``None``.
    normalize : bool, optional
        If ``True``, expression per group is divided by the global mean across all
        bins (fold-change over background).  If ``False`` (default), returns the
        fraction of bins in each group that express each gene.

    Returns
    -------
    ad.AnnData
        The input ``adata`` updated in-place with:

        * ``uns[f'interface_{key}_profile']`` – DataFrame (genes × groups)
    """
    array_usr = (adata.uns['binned_data'].toarray().reshape(adata.uns['binned_data_shape']) * adata.uns["mask_ecm"][:, :, None]).copy()
    genes = adata.uns['genes']

    if key not in adata.obs:
        raise ValueError(f"adata.obs must contain '{key}' column for key '{key}'")
    cell_types = adata.obs[key].to_numpy(dtype=object)
    unique_types = np.unique(cell_types)

    target_coords_xy = np.argwhere(adata.uns["interface_map"] == 1)
    inverted_dict = {t: [] for t in unique_types}
    knn_idx = adata.uns["interface_knn_idx"]
    for i, bins in enumerate(knn_idx):
        closest_types = cell_types[bins]
        for t in closest_types:
            inverted_dict[t].append(tuple(target_coords_xy[i]))

    if normalize:
        H, W, n_genes = array_usr.shape
        global_mean = array_usr.reshape(-1, n_genes).mean(axis=0)
        global_mean = np.where(global_mean == 0, 1.0, global_mean)

    def profile_dict(coord_dict):
        result = {}
        for t, v in coord_dict.items():
            if len(v) == 0:
                continue
            rows, cols = zip(*v)
            rows = np.array(rows)
            cols = np.array(cols)
            values = array_usr[rows, cols, :]  # shape (len(v), n_genes)
            if normalize:
                result[t] = values.mean(axis=0) / global_mean
            else:
                binary = (values > 0).astype(float)
                result[t] = binary.mean(axis=0)
        return result

    s_dict = profile_dict(inverted_dict)
    s_df = pd.DataFrame.from_dict(s_dict, orient="index", columns=genes)
    s_df = s_df.loc[:, lambda df: df.sum() != 0]

    combined = list(set(adata.uns['axon_markers']).union(adata.uns['dendrite_markers']))
    s_df = s_df.drop(columns=combined, errors="ignore")
    s_df = s_df.loc[:, s_df.sum() != 0]

    suffix = "_normalized" if normalize else ""
    adata.uns[f"interface_{key}_profile{suffix}"] = s_df.T
    return adata

def communication_profile(adata, key="cell_type", k=1, threshold=None, device='cpu'):
    """
    Build a ligand-receptor communication profile grouped by a cell-level annotation.

    For each significant LR pair (from ``adata.uns['cellwhisper_lr']``), computes the
    spatial edge map (axon ligand → dendrite receptor co-localisation), finds the
    midpoint of each edge, and assigns it to the nearest cell's annotation label.
    The result is a count table of LR edges per group.

    Parameters
    ----------
    adata : ad.AnnData
        Must contain ``adata.obs[key]``, ``adata.uns['cellwhisper_lr']``,
        ``adata.uns['bin_probabilities']``, ``adata.uns['lr_pairs']``,
        ``adata.uns['binned_data']``, ``adata.uns['binned_data_shape']``,
        ``adata.uns['mask_ecm']``, ``adata.uns['mask_cell']``,
        ``adata.uns['genes']``, and ``adata.obsm['bin']``.
    key : str, optional
        Column in ``adata.obs`` used to group LR edges.
        Default: ``'cell_type'``.
    k : int, optional
        Number of nearest-neighbour cells to consider per edge midpoint.
        Default: 1.
    threshold : float, optional
        Probability threshold for binarising axon/dendrite maps.
        Default: ``None`` (treated as 0.5 internally).
    device : str, optional
        Torch device string, e.g. ``'cpu'`` or ``'cuda'``.  Default: ``'cpu'``.

    Returns
    -------
    ad.AnnData
        The input ``adata`` updated in-place with:

        * ``uns['lr_edges']``                    – dict mapping LR pair string to list of edges
        * ``uns[f'communication_{key}_profile']`` – DataFrame (groups × LR pairs) of edge counts
    """
    if key not in adata.obs:
        raise ValueError(f"adata.obs must contain '{key}' column for mode '{key}'")
    cell_types = adata.obs[key].to_numpy(dtype=object)
    unique_types = np.unique(cell_types)

    array_usr = (adata.uns['binned_data'].toarray().reshape(adata.uns['binned_data_shape']) * adata.uns["mask_ecm"][:, :, None]).copy()
    genes = list(adata.uns['genes'])
    # # remove later
    # with open("../SCOUBI/scoubi/data/pairs.pkl", "rb") as fp:
    #     pairs = pickle.load(fp)
    # pairs = [pair for pair in pairs if pair[0] in genes and pair[1] in genes]
    # #--------------
    pairs = adata.uns['lr_pairs'] 
    ad_map = torch.from_numpy(adata.uns['bin_probabilities'].copy()).float().to(device)
    binary_matrix_ecm = torch.from_numpy(adata.uns['mask_ecm'].copy()).float().to(device)
    binary_matrix_cell = torch.from_numpy(adata.uns['mask_cell'].copy()).float().to(device)
    binary_overlap = (binary_matrix_cell * binary_matrix_ecm).cpu().numpy()
    x_bin, x_shape, gene_to_idx = _prep_dict(array_usr, pairs, genes, device)
    threshold = threshold if threshold is not None else 0.5
    ad_map[ad_map <= threshold] = 0
    ad_map[ad_map > threshold] = 1
    significant_lr_pairs = adata.uns['cellwhisper_lr']
    lr_edges = {}
    lr_edges_end = {}
    for gp in significant_lr_pairs:
        a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
        if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin: continue
        x_a, x_b = x_bin[a_idx], x_bin[b_idx]
        X = get_whisper_edges(ad_map[:, 0].reshape(binary_matrix_ecm.shape), ad_map[:, 1].reshape(binary_matrix_ecm.shape), x_a, x_b, device)
        X[X > 0] = 1
        lr_edges[tuple(gp)] = X.cpu().numpy()
        X = get_whisper_edges(ad_map[:, 1].reshape(binary_matrix_ecm.shape), ad_map[:, 0].reshape(binary_matrix_ecm.shape), x_b, x_a, device)
        X[X > 0] = 1
        lr_edges_end[tuple(gp)] = X.cpu().numpy()
    edges = {}
    for lr, mat_start in lr_edges.items():
        mat_end = lr_edges_end.get(lr, [])
        starts = np.argwhere(mat_start == 1)
        ends = np.argwhere(mat_end == 1)
        edges_lr = set()
        for start in starts:
            linked_ends = [end for end in ends if max(abs(end[0]-start[0]), abs(end[1]-start[1])) == 1]
            for end in linked_ends:
                edge = tuple(sorted([tuple(start), tuple(end)]))
                edges_lr.add(edge)
        edges["_".join(lr)] = list(edges_lr)

    count_table = pd.DataFrame(0, index=unique_types, columns=["_".join(x) for x in significant_lr_pairs])
    tree = cKDTree(adata.obsm['bin'])

    for lr, edge_list in edges.items():
        all_centers = []
        for edge in edge_list:
            p1, p2 = np.array(edge[0]), np.array(edge[1])
            center = (p1 + p2) / 2.0
            all_centers.append(center)
        dists, idxs = tree.query(all_centers, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]
        closest_types = cell_types[idxs].ravel()
        counts = pd.Series(closest_types).value_counts()
        counts = counts.reindex(count_table.index, fill_value=0)
        count_table[lr] = count_table[lr].add(counts)

    adata.uns['lr_edges'] = edges
    adata.uns[f'communication_{key}_profile'] = count_table
    return adata
