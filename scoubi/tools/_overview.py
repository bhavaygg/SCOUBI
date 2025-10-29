import numpy as np
import anndata as ad
import pandas as pd
import torch
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ..model import do_conv, _prep_dict, kernel
import pickle

def run_cellwhisper(Z_1, Z_2, x_a, x_b, device, eps = 1e-30):
    agg_Z_2 = do_conv(Z_2, kernel=kernel.to(device))
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel=kernel.to(device))
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
    return z_score.item(), N.item(), X.item(), p_a.item(), p_d.item()

def get_whisper_edges(Z_1, Z_2, x_a, x_b, device):
    agg_Z_2_x_b = do_conv(Z_2 * x_b, kernel=kernel.to(device))
    X = (Z_1 * x_a) * agg_Z_2_x_b
    return X

def fmt(val, total_bins_val):
    pct = val / total_bins_val * 100
    return f"{val:,} ({pct:.1f}%)"

def probability_histograms(ad_map, s_map_binary_a, s_map_binary_d, binary_matrix_ecm,
                           bin_start=0.5, bin_end=1.0, bin_width=0.1, width=20):
    probs = {
        "Axonic": ad_map[:, 0].cpu().numpy(),
        "Dendritic": ad_map[:, 1].cpu().numpy(),
        "Presynaptic": (ad_map[:, 0].cpu().numpy().reshape(s_map_binary_a.shape) * s_map_binary_a).flatten(),
        "Postsynaptic": (ad_map[:, 1].cpu().numpy().reshape(s_map_binary_d.shape) * s_map_binary_d).flatten()
    }
    mask = binary_matrix_ecm.cpu().numpy().flatten() > 0

    histograms = {}
    bins = np.arange(bin_start, bin_end + 1e-8, bin_width)
    for name, p in probs.items():
        # p = p[mask]
        p = p[p >= bin_start]
        hist, bin_edges = np.histogram(p, bins=bins)
        max_val = hist.max() if hist.max() > 0 else 1

        hist_str = ""
        for i in range(len(hist)):
            bar = "█" * int(hist[i] / max_val * width)
            hist_str += f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}: {bar} ({hist[i]})\n"
        histograms[name] = hist_str.strip()
    return histograms

def overview(adata: ad.AnnData, threshold = None, zscore_threshold = 3, cw_edges_threshold = 30, device = 'cpu'):
    array_usr = (adata.uns['binned_data'] * adata.uns["mask_ecm"][:, :, None]).copy()
    genes = list(adata.uns['genes'])
    # # remove later
    # with open("../SCOUBI/scoubi/data/pairs.pkl", "rb") as fp:
    #     pairs = pickle.load(fp)
    # pairs = [pair for pair in pairs if pair[0] in genes and pair[1] in genes]
    # #--------------
    pairs = adata.uns['lr_pairs'] 
    genes = adata.uns['genes']
    ad_map = torch.from_numpy(adata.uns['bin_probabilities'].copy()).float().to(device)
    binary_matrix_ecm = torch.from_numpy(adata.uns['mask_ecm'].copy()).float().to(device)
    binary_matrix_cell = torch.from_numpy(adata.uns['mask_cell'].copy()).float().to(device)
    binary_overlap = (binary_matrix_cell * binary_matrix_ecm).cpu().numpy()
    x_bin, x_shape, gene_to_idx = _prep_dict(array_usr, pairs, genes, device)
    threshold = threshold if threshold is not None else 0.5
    # adata.uns['threshold'] = threshold
    ad_map[ad_map <= threshold] = 0
    ad_map[ad_map > threshold] = 1
    rows = []
    for gp in pairs:
        a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
        if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin: continue
        x_a, x_b = x_bin[a_idx], x_bin[b_idx]
        zscore, N, X, p_a, p_d = run_cellwhisper(ad_map[:, 0].reshape(binary_matrix_ecm.shape), ad_map[:, 1].reshape(binary_matrix_ecm.shape), x_a, x_b, device)
        rows.append([gp[0], gp[1], zscore, N, X, p_a, p_d])
    df_cw = pd.DataFrame(rows, columns = ['L', 'R', 'zscore', 'N', 'X', 'p_a', 'p_d'])
    significant_pairs = df_cw[(df_cw.zscore >= zscore_threshold) & (df_cw.X >= cw_edges_threshold)].copy()
    significant_lr_pairs = list(set((row.L, row.R) for _, row in significant_pairs.iterrows()))
    adata.uns['cellwhisper'] = significant_pairs
    adata.uns['cellwhisper_lr'] = significant_lr_pairs
    s_map_a = torch.zeros_like(binary_matrix_ecm)
    s_map_d = torch.zeros_like(binary_matrix_ecm)
    for gp in significant_lr_pairs:
        a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
        if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin: continue
        x_a, x_b = x_bin[a_idx], x_bin[b_idx]
        X = get_whisper_edges(ad_map[:, 0].reshape(binary_matrix_ecm.shape), ad_map[:, 1].reshape(binary_matrix_ecm.shape), x_a, x_b, device)
        s_map_a += X
        X = get_whisper_edges(ad_map[:, 1].reshape(binary_matrix_ecm.shape), ad_map[:, 0].reshape(binary_matrix_ecm.shape), x_b, x_a, device)
        s_map_d += X
    s_map_binary_a = (s_map_a > 0).float().cpu().numpy()
    s_map_binary_d = (s_map_d > 0).float().cpu().numpy()
    a_map = (ad_map[:, 0].reshape(binary_matrix_ecm.shape).clone() > 0).float().cpu().numpy()
    d_map = (ad_map[:, 1].reshape(binary_matrix_ecm.shape).clone() > 0).float().cpu().numpy()
    total_bins = int(binary_matrix_ecm.shape[0] * binary_matrix_ecm.shape[1])
    usr_bins = int(binary_matrix_ecm.sum().item() - binary_overlap.sum())
    cell_bins = int(binary_matrix_cell.sum().item() - binary_overlap.sum())
    overlap_bins = int(binary_overlap.sum())
    axonic_bins = int(a_map.sum())
    dendritic_bins = int(d_map.sum())
    presynaptic_bins = int(s_map_binary_a.sum())
    postsynaptic_bins = int(s_map_binary_d.sum())
    ad_map = torch.from_numpy(adata.uns['bin_probabilities'].copy()).float().to(device)
    ad_map[ad_map <= threshold] = 0

    console = Console()
    table = Table(show_header=False, box=None)

    table.add_row("[bold pink1]Total bins[/bold pink1]", f"[bold pink1]{total_bins:,}[/bold pink1]")  # no percentage
    table.add_row("[bold]USR bins[/bold]", f"[bold]{fmt(usr_bins, total_bins)}[/bold]")
    table.add_row("[bold]Cell bins[/bold]", f"[bold]{fmt(cell_bins, total_bins)}[/bold]")
    table.add_row("[bold]Overlap bins[/bold]", f"[bold]{fmt(overlap_bins, total_bins)}[/bold]")
    table.add_row("[bold deep_pink2]Axonic bins[/bold deep_pink2]", f"[bold deep_pink2]{fmt(axonic_bins, total_bins)}[/bold deep_pink2]")
    table.add_row("[bold sky_blue1]Dendritic bins[/bold sky_blue1]", f"[bold sky_blue1]{fmt(dendritic_bins, total_bins)}[/bold sky_blue1]")
    table.add_row("[bold orange1]Presynaptic bins[/bold orange1]", f"[bold orange1]{fmt(presynaptic_bins, total_bins)}[/bold orange1]")
    table.add_row("[bold spring_green1]Postsynaptic bins[/bold spring_green1]", f"[bold spring_green1]{fmt(postsynaptic_bins, total_bins)}[/bold spring_green1]")
    histograms = probability_histograms(ad_map, s_map_binary_a, s_map_binary_d, binary_matrix_ecm)

    table.add_row("[bold deep_pink2]Axonic Probabilities[/bold deep_pink2]", f"[bold deep_pink2]{histograms['Axonic']}[/bold deep_pink2]")
    table.add_row("[bold sky_blue1]Dendritic Probabilities[/bold sky_blue1]", f"[bold sky_blue1]{histograms['Dendritic']}[/bold sky_blue1]")
    table.add_row("[bold orange1]Presynaptic Probabilities[/bold orange1]", f"[bold orange1]{histograms['Presynaptic']}[/bold orange1]")
    table.add_row("[bold spring_green1]Postsynaptic Probabilities[/bold spring_green1]", f"[bold spring_green1]{histograms['Postsynaptic']}[/bold spring_green1]")
    panel = Panel(table, style="on grey15", padding=(1,2), title="Overview", border_style="grey70")
    console.print(table)

    adata.uns['axon_map'] = a_map
    adata.uns['dendrite_map'] = d_map
    adata.uns['synapse_map'] = s_map_binary_d + s_map_binary_a
    adata.uns['presynapse_map'] = s_map_binary_a
    adata.uns['postsynapse_map'] = s_map_binary_d
    adata.uns['data_shape'] = list(x_shape)
    return adata