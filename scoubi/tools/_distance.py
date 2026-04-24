import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def distance(adata, k = 1):
    tree = cKDTree(adata.obsm['bin'])

    # for key in ['presynapse_map', 'postsynapse_map']:
    key = "interface_map"
    target_coords_xy = np.argwhere(adata.uns[key] == 1)
    dists, idxs = tree.query(target_coords_xy, k=k)
    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]
    prefix = key.split("_")[0]
    adata.uns[f"{prefix}_knn_dists"] = dists.astype(np.float32)
    adata.uns[f"{prefix}_knn_idx"] = idxs

    return adata
