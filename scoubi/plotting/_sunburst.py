import numpy as np
import anndata as ad
import pandas as pd
import torch
import matplotlib.pyplot as plt
from urllib3.util.retry import Retry
from matplotlib.patches import Wedge
import matplotlib.cm as cm
from ..model import get_zscore, do_conv, _prep_dict, kernel
from ..tools import get_whisper_edges
import pickle

def draw_wedge(center, r_in, r_out, start, end, color, ax, edge='black', linewidth = 0.5, hatch=None, dash=False):
    w = Wedge(center, r_out, start, end, width=r_out-r_in,
              facecolor=color, edgecolor=edge,
              linestyle='-', linewidth=linewidth, hatch=hatch)
    ax.add_patch(w)

def draw_gradient_wedge(center, r_in, r_out, start_angle, total_span, values, colors, ax, linewidth = 0.5, hatch=None, dash=False):
    total = sum(values)
    current_angle = start_angle
    for n, v in enumerate(values):
        span = total_span * v / total
        draw_wedge(center, r_in, r_out, current_angle, current_angle+span, colors[n], ax, linewidth = linewidth, hatch=hatch, dash=dash)
        current_angle += span

def sunburst(adata: ad.AnnData, threshold = None, device = 'cpu'):
    array_usr = (adata.uns['binned_data'] * adata.uns["mask_ecm"][:, :, None]).copy()
    genes = list(adata.uns['genes'])
    # remove later
    with open("../SCOUBI/scoubi/data/pairs.pkl", "rb") as fp:
        pairs = pickle.load(fp)
    pairs = [pair for pair in pairs if pair[0] in genes and pair[1] in genes]
    #--------------
    # pairs = adata.uns['pairs'] #unomment later
    genes = adata.uns['genes'].tolist()
    ad_map = torch.from_numpy(adata.uns['bin_probabilities']).float().to(device)
    binary_matrix_ecm = torch.from_numpy(adata.uns['mask_ecm']).float().to(device)
    binary_matrix_cell = torch.from_numpy(adata.uns['mask_cell']).float().to(device)
    binary_overlap = (binary_matrix_cell * binary_matrix_ecm).cpu().numpy()
    threshold = threshold if threshold is not None else 0.5
    # adata.uns['threshold'] = threshold
    a_map = adata.uns['axon_map']
    d_map = adata.uns['dendrite_map']
    presynapse_map = adata.uns['presynapse_map']
    postsynapse_map = adata.uns['postsynapse_map']
    total_bins = int(binary_matrix_ecm.shape[0] * binary_matrix_ecm.shape[1])
    usr_bins = int(binary_matrix_ecm.sum().item() - binary_overlap.sum())
    cell_bins = int(binary_matrix_cell.sum().item() - binary_overlap.sum())
    overlap_bins = int(binary_overlap.sum())
    axonic_bins = int(a_map.sum())
    dendritic_bins = int(d_map.sum())
    presynaptic_bins = int(presynapse_map.sum())
    postsynaptic_bins = int(postsynapse_map.sum())
    none_bins = total_bins - (usr_bins + cell_bins + overlap_bins)
    ad_map = torch.from_numpy(adata.uns['bin_probabilities']).float().to(device)
    ad_map[ad_map <= threshold] = 0
    bin_edges = np.arange(threshold, 1.01, 0.1)
    bin_counts_a = []
    bin_counts_sa = []
    bin_counts_d = []
    bin_counts_sd = []
    a_map_raw = ad_map[:, 0].clone().cpu().numpy()
    d_map_raw = ad_map[:, 1].clone().cpu().numpy()
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i+1]
        count = ((a_map_raw > low) & (a_map_raw <= high)).sum()
        temp = np.zeros_like(a_map_raw)
        bin_counts_a.append(count)
        temp[(a_map_raw > low) & (a_map_raw <= high)] = 1
        bin_counts_sa.append((presynapse_map.flatten() * temp).sum().astype(int))

        low, high = bin_edges[i], bin_edges[i+1]
        count = ((d_map_raw > low) & (d_map_raw <= high)).sum()
        bin_counts_d.append(count)
        temp = np.zeros_like(d_map_raw)
        temp[(d_map_raw > low) & (d_map_raw <= high)] = 1
        bin_counts_sd.append((postsynapse_map.flatten() * temp).sum().astype(int))

    fig, ax = plt.subplots(figsize=(10,10))
    draw_wedge((0,0), 0, 0.4, 0, 360, '#fff0ff', ax, linewidth = 1)  # Tissue
    # USR, Both, Cell
    usr_start = 0
    usr_span = 360 * usr_bins / total_bins
    both_start = usr_start + usr_span
    both_span = 360 * overlap_bins / total_bins
    none_start = both_start + both_span
    none_span = 360 * none_bins / total_bins
    cell_start = none_start + none_span
    cell_span = 360 * cell_bins / total_bins


    draw_wedge((0,0), 0.4, 0.65, usr_start, usr_start+usr_span, 'white', ax, linewidth = 1)
    draw_wedge((0,0), 0.4, 0.65, both_start, both_start+both_span, 'white', ax, hatch = "XX", dash=True, edge="black", linewidth = 1)
    draw_wedge((0,0), 0.4, 0.65, none_start, none_start+none_span, 'white', ax, linewidth = 0, edge=None)
    draw_wedge((0,0), 0.4, 0.65, cell_start, cell_start+cell_span, 'black', ax, linewidth = 1)

    # --- Hierarchical wedges with gradients ---
    draw_gradient_wedge((0,0), 0.65, .9, usr_start, usr_span * axonic_bins / usr_bins, bin_counts_a, ['#660000', '#990000','#cc0000', '#ff4d4d', '#ff9999', ][::-1], ax, linewidth=0.5) # Axonic
    draw_gradient_wedge((0,0), 0.65, .9, usr_start + usr_span * axonic_bins / usr_bins, usr_span * dendritic_bins / usr_bins, bin_counts_d, ['#00264d', '#003366', '#004d99', '#66b3ff', '#99ccff'][::-1], ax, linewidth=0.5)  # Dendritic
    draw_gradient_wedge((0,0), .9, 1.2, usr_start, usr_span * presynaptic_bins / usr_bins, bin_counts_sa, ['#993d00', '#cc5200', '#ff6600', '#ff8533', '#ffb380'][::-1], ax, linewidth=0.5)  # Presynaptic
    draw_gradient_wedge((0,0), .9, 1.2, usr_start + usr_span * axonic_bins / usr_bins, usr_span * postsynaptic_bins / usr_bins, bin_counts_sd, ['#145214', '#1f7a1f', '#29a329', '#47d147', '#99e699'][::-1], ax, linewidth=0.5)  # Postsynaptic

    ax.set(aspect='equal')
    ax.set_xlim(-1.8,1.8)
    ax.set_ylim(-1.8,1.8)
    plt.axis("off")
    plt.show()