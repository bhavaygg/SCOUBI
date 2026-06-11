import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def distance(adata, k=1):
    """
    Assign every interface bin to its nearest cell(s) via k-NN on bin coordinates.

    Builds a KD-tree over ``adata.obsm['bin']`` (cell bin-coordinates set by
    :func:`scoubi.pp.bin_data`) and queries each pixel in ``adata.uns['interface_map']``
    for its ``k`` nearest cells.  Results are stored in ``adata.uns`` under the
    ``interface_`` prefix.

    Parameters
    ----------
    adata : ad.AnnData
        Must contain ``adata.obsm['bin']`` and ``adata.uns['interface_map']``.
    k : int, optional
        Number of nearest neighbours to retrieve per interface pixel.  Default: 1.

    Returns
    -------
    ad.AnnData
        The input ``adata`` updated in-place with:

        * ``uns['interface_knn_dists']`` – float32 array (n_pixels × k) of distances
        * ``uns['interface_knn_idx']``   – int array (n_pixels × k) of cell indices
    """
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
