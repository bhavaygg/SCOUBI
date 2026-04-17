import numpy as np
import anndata as ad
import pandas as pd
import torch
from ..model import do_conv, _prep_dict, kernel
from scipy.spatial import cKDTree

def run_cellwhisper(Z_1, Z_2, x_a, x_b, device, eps = 1e-30):
    agg_Z_2 = do_conv(Z_2, kernel=kernel.to(device))
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel=kernel.to(device))
    X_raw = torch.sum(x_a * do_conv(x_b, kernel=kernel.to(device)))
    X = (Z_1 * x_a) * agg_Z_2_x_b
    Z_prod = Z_1 * agg_Z_2
    N = torch.sum(Z_prod)
    X_mat = (Z_1 * x_a) * agg_Z_2_x_b
    X = torch.sum(X_mat)
    p_a = torch.sum(x_a * Z_1) / (torch.sum(Z_1) + eps)
    p_d = torch.sum(x_b * Z_2) / (torch.sum(Z_2) + eps)
    E_x = N * p_a * p_d
    E_x_2 = torch.sum((Z_prod * p_a * p_d)**2) + (torch.sum(Z_prod)**2 - torch.sum(Z_prod**2)) * (p_a * p_d)**2
    var_X = E_x + E_x_2 - E_x**2
    var_X = torch.clamp(var_X, min=eps)
    z_score = (X - E_x) / torch.sqrt(var_X)
    return z_score.item(), N.item(), X.item(), p_a.item(), p_d.item(), X_raw.item()

def get_whisper_edges(Z_1, Z_2, x_a, x_b, device):
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel=kernel.to(device))
    X = (Z_1 * x_a) * agg_Z_2_x_b
    return X

def run_cw_regionwise(adata: ad.AnnData, threshold = None, zscore_threshold = 3, cw_edges_threshold = 30, device = 'cpu'):
    array_usr = (adata.uns['binned_data'].toarray().reshape(adata.uns['binned_data_shape']) * adata.uns["mask_ecm"][:, :, None]).copy()
    pairs = adata.uns['lr_pairs'] 
    genes = adata.uns['genes']
    ad_map = torch.from_numpy(adata.uns['bin_probabilities'].copy()).float().to(device)
    binary_matrix_ecm = torch.from_numpy(adata.uns['mask_ecm'].copy()).float().to(device)
    binary_matrix_cell = torch.from_numpy(adata.uns['mask_cell'].copy()).float().to(device)
    threshold = threshold if threshold is not None else 0.5
    # adata.uns['threshold'] = threshold
    ad_map[ad_map <= threshold] = 0
    ad_map[ad_map > threshold] = 1

    tree = cKDTree(adata.obsm['bin'])
    k = 1
    coords = np.ones_like(adata.uns['interface_map'])
    target_coords_xy = np.argwhere(coords == 1)
    dists, idxs = tree.query(target_coords_xy, k=1)
    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]
    regions = adata.obs['region'].values
    unique_types = np.unique(adata.obs['region'].values)
    inverted_dict = {t: [] for t in unique_types}
    for i, bins in enumerate(idxs):
        closest_types = regions[bins]
        for t in closest_types:
            inverted_dict[t].append(tuple(target_coords_xy[i]))
    region_wise_results = {}
    for region in np.unique(regions):
        print(region)
        temp = np.zeros_like(adata.uns['interface_map'])
        coords = np.array(inverted_dict[region])
        x, y = coords[:, 0], coords[:, 1]
        temp[x, y] = 1
        array_usr_temp = array_usr.copy() * temp[:, :, np.newaxis]
        x_bin, x_shape, gene_to_idx = _prep_dict(array_usr_temp, pairs, genes, device)
        temp = torch.from_numpy(temp).float().to(device)
        ad_map_region = ad_map.clone()
        rows = []
        for gp in pairs:
            a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
            if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin: continue
            x_a, x_b = x_bin[a_idx], x_bin[b_idx]
            zscore, N, X, p_a, p_d, X_raw = run_cellwhisper(ad_map_region[:, 0].reshape(binary_matrix_ecm.shape) * temp , ad_map_region[:, 1].reshape(binary_matrix_ecm.shape) * temp, x_a, x_b, device)
            rows.append([gp[0], gp[1], zscore, N, X, p_a, p_d, X_raw])
        df_cw = pd.DataFrame(rows, columns = ['L', 'R', 'zscore', 'N', 'X', 'p_a', 'p_d', 'X_neighboring'])
        significant_pairs = df_cw[(df_cw.zscore >= zscore_threshold) & (df_cw.X >= cw_edges_threshold)].copy()
        region_wise_results[region] = significant_pairs
    adata.uns['cellwhisper_regionwise'] = region_wise_results
    return adata