import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# def distance(adata, threshold = 0.1, k = 1):
#     array_usr = adata.uns["binned_data"]
#     genes = adata.uns['genes']
#     coords = []

#     bin_coords = adata.obsm['bin']
#     cell_types = adata.obs['cell_type'].values
#     unique_types = np.unique(cell_types)

#     for n, s_map in enumerate([adata.uns['presynapse_map'], adata.uns['postsynapse_map']]):
#         target_coords_xy = np.argwhere(s_map == 1)
#         dist_matrix = cdist(target_coords_xy, bin_coords, "euclidean")

#         if n == 0:
#             adata.uns["presynapse_distances"] = dist_matrix
#         else:
#             adata.uns["postsynapse_distances"] = dist_matrix

#         inverted_dict = {t: [] for t in unique_types}

#         for i in range(dist_matrix.shape[0]):
#             dists = dist_matrix[i]
#             k_closest_idx = np.argsort(dists)[:k]
#             closest_types = cell_types[k_closest_idx]
#             for t in closest_types:
#                 inverted_dict[t].append(tuple(target_coords_xy[i]))

#         coords.append(inverted_dict)

#     def profile_dict(coord_dict):
#         return {
#             k: array_usr[np.array([r for r, _ in v]),
#                          np.array([c for _, c in v]), :].mean(0)
#             for k, v in coord_dict.items() if len(v) > 0
#         }

#     a_dict = profile_dict(coords[0])
#     d_dict = profile_dict(coords[1])

#     a_df = pd.DataFrame.from_dict(a_dict, orient="index", columns=genes)
#     d_df = pd.DataFrame.from_dict(d_dict, orient="index", columns=genes)

#     a_df[a_df < threshold] = 0
#     d_df[d_df < threshold] = 0

#     s_df = (a_df + d_df).loc[:, lambda df: df.sum() != 0]

#     combined = set(adata.uns['axon_markers']).union(adata.uns['dendrite_markers'])
#     s_df = s_df.drop(columns=combined, errors="ignore")
#     a_df = a_df.loc[:, a_df.sum() != 0]
#     d_df = d_df.loc[:, d_df.sum() != 0]

#     adata.uns["presynapse_profile"] = a_df
#     adata.uns["postsynapse_profile"] = d_df
#     adata.uns["synapse_profile"] = s_df

#     return adata


def distance(adata, k = 1):
    tree = cKDTree(adata.obsm['bin'])

    for key in ['presynapse_map', 'postsynapse_map']:
        target_coords_xy = np.argwhere(adata.uns[key] == 1)
        dists, idxs = tree.query(target_coords_xy, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]
        adata.uns[f"{key.split("_")[0]}_knn_dists"] = dists.astype(np.float32)
        adata.uns[f"{key.split("_")[0]}_knn_idx"] = idxs 

    return adata