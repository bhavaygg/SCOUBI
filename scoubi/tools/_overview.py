import numpy as np
import anndata as ad
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import torch.nn.functional as F
from scipy.stats import binom
from ..model import do_conv, _prep_dict, kernel

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

def p_geq_k(p: float, k: int, n: int = 4) -> float:
    return 1 - binom.cdf(k - 1, n, p)

def fmt(val, total_bins_val):
    pct = val / total_bins_val * 100
    return f"{val:,} ({pct:.1f}%)"

def get_interfaces(Z_1, Z_2, x_a, x_b, device, kernel_size=2, filter=(2, 1)):
    """
    Detect interfaces.

    Parameters
    ----------
    kernel_size : int
        Side length N of the NxN sliding window used to detect interfaces.
        A block qualifies if it contains >= N cells from each population.
    filter : tuple[int, int]
        Expression thresholds (strict, lenient).  A block passes if
        (xa_count >= strict AND xb_count >= strict) OR
        (xa_count >= lenient AND xb_count >= lenient).
    """
    k = torch.ones((1, 1, kernel_size, kernel_size), device=device)
    def sum_nxn(X):
        return F.conv2d(X[None, None], k)[0, 0]
    cond_split = (sum_nxn(Z_1) >= kernel_size) & (sum_nxn(Z_2) >= kernel_size)
    xa_cnt = sum_nxn(x_a)
    xb_cnt = sum_nxn(x_b)
    cond_tx = ((xa_cnt >= filter[0]) & (xb_cnt >= filter[1])) | ((xa_cnt >= filter[1]) & (xb_cnt >= filter[0]))
    blocks = (cond_split & cond_tx).float()
    pixels = F.conv_transpose2d(
        blocks[None, None],
        k,
        stride=1
    )[0, 0]
    return (pixels > 0).int(), blocks.sum()

def probability_histograms(ad_map, s_map_binary, binary_matrix_ecm,
                           bin_start=0.5, bin_end=1.0, bin_width=0.1, width=20):
    probs = {
        "Axonic": ad_map[:, 0].cpu().numpy(),
        "Dendritic": ad_map[:, 1].cpu().numpy(),
        "Interface": (ad_map[:, 0].cpu().numpy().reshape(s_map_binary.shape) * s_map_binary).flatten() + (ad_map[:, 1].cpu().numpy().reshape(s_map_binary.shape) * s_map_binary).flatten(),
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

def overview(
    adata: ad.AnnData,
    threshold=None,
    zscore_threshold=3,
    cw_edges_threshold=30,
    device='cpu',
    kernel_size=2,
    filter=(2, 1),
):
    """
    Run the full CellWhisper pipeline and print a spatial overview.

    Computes ligand-receptor communication scores for all pairs, identifies
    statistically significant interactions, detects axon-dendrite interface
    regions, and prints a summary table with spatial bin counts and
    probability histograms.

    Results are stored in ``adata.uns``:

    * ``cellwhisper``            – significant LR pairs (filtered DataFrame)
    * ``cellwhisper_unfiltered`` – scores for all tested LR pairs
    * ``cellwhisper_lr``         – unique significant (L, R) tuples
    * ``axon_map``               – binary axonic bin map (H x W)
    * ``dendrite_map``           – binary dendritic bin map (H x W)
    * ``interface_map``          – binary interface bin map (H x W)
    * ``data_shape``             – spatial grid dimensions [H, W]

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.  Must contain the keys populated by
        ``scoubi.model.train``: ``binned_data``, ``binned_data_shape``,
        ``mask_ecm``, ``mask_cell``, ``bin_probabilities``, ``lr_pairs``,
        and ``genes``.
    threshold : float, optional
        Probability threshold for binarising axon/dendrite maps.
        Defaults to 0.5.
    zscore_threshold : float
        Minimum CellWhisper z-score for a pair to be considered
        significant.  Default: 3.
    cw_edges_threshold : float
        Minimum number of co-localised edges (X) for a pair to be
        considered significant.  Default: 30.
    device : str
        Torch device string, e.g. ``'cpu'`` or ``'cuda'``.  Default: ``'cpu'``.
    kernel_size : int
        Side length N of the NxN sliding window used by
        :func:`get_interfaces` to detect interface blocks.  Default: 2.
    filter : tuple[int, int]
        Expression count thresholds ``(strict, lenient)`` passed to
        :func:`get_interfaces`.  A block passes if both genes meet the
        strict threshold, or both meet the lenient threshold.  Default: (2, 1).

    Returns
    -------
    ad.AnnData
        The input ``adata`` with results written to ``adata.uns``.
    """
    array_usr = (adata.uns['binned_data'].toarray().reshape(adata.uns['binned_data_shape']) * adata.uns["mask_ecm"][:, :, None]).copy()
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
        total_bins_val = binary_matrix_ecm.shape[0] * binary_matrix_ecm.shape[1]
        p_a_global = (x_bin[a_idx].sum() / total_bins_val).item()
        p_b_global = (x_bin[b_idx].sum() / total_bins_val).item()
        n_bins = kernel_size ** 2
        strict, lenient = filter
        p_null = (p_geq_k(p_a_global, strict, n_bins) * p_geq_k(p_b_global, lenient, n_bins)) + \
                (p_geq_k(p_a_global, lenient, n_bins) * p_geq_k(p_b_global, strict, n_bins)) - \
                (p_geq_k(p_a_global, strict, n_bins) * p_geq_k(p_b_global, strict, n_bins))
        rows.append([gp[0], gp[1], zscore, N, X, p_a, p_d, p_a_global, p_b_global, p_null])
    df_cw = pd.DataFrame(rows, columns=['L', 'R', 'zscore', 'N', 'X', 'p_a', 'p_d', 'p_l_global', 'p_r_global', 'p_null_joint'])
    significant_pairs = df_cw[(df_cw.zscore >= zscore_threshold) & (df_cw.X >= cw_edges_threshold)].copy()
    significant_lr_pairs = list(set((row.L, row.R) for _, row in significant_pairs.iterrows()))
    adata.uns['cellwhisper'] = significant_pairs
    adata.uns['cellwhisper_unfiltered'] = df_cw
    adata.uns['cellwhisper_lr'] = significant_lr_pairs
    s_map = torch.zeros_like(binary_matrix_ecm)
    s_count = 0
    for gp in significant_lr_pairs:
        a_idx, b_idx = gene_to_idx.get(gp[0]), gene_to_idx.get(gp[1])
        if a_idx is None or b_idx is None or a_idx not in x_bin or b_idx not in x_bin: continue
        x_a, x_b = x_bin[a_idx], x_bin[b_idx]
        pixel_map, _ = get_interfaces(ad_map[:, 0].reshape(binary_matrix_ecm.shape), ad_map[:, 1].reshape(binary_matrix_ecm.shape), x_a, x_b, device, kernel_size=kernel_size, filter=filter)
        s_map += pixel_map
        s_count += _
    s_map_binary = (s_map > 0).float().cpu().numpy()
    a_map = (ad_map[:, 0].reshape(binary_matrix_ecm.shape).clone() > 0).float().cpu().numpy()
    d_map = (ad_map[:, 1].reshape(binary_matrix_ecm.shape).clone() > 0).float().cpu().numpy()
    total_bins = int(binary_matrix_ecm.shape[0] * binary_matrix_ecm.shape[1])
    usr_bins = int(binary_matrix_ecm.sum().item() - binary_overlap.sum())
    cell_bins = int(binary_matrix_cell.sum().item() - binary_overlap.sum())
    overlap_bins = int(binary_overlap.sum())
    axonic_bins = int(a_map.sum())
    dendritic_bins = int(d_map.sum())
    interface_bins = int(s_map_binary.sum())
    n_interfaces = int(s_count)
    ad_map = torch.from_numpy(adata.uns['bin_probabilities'].copy()).float().to(device)
    ad_map[ad_map <= threshold] = 0
    n_sig = len(significant_pairs)
    n_tested = len(df_cw)
    n_sig_null = (significant_pairs['p_null_joint'] < 0.05).sum()
    max_null = significant_pairs['p_null_joint'].max()


    console = Console()
    table = Table(show_header=False, box=None)
    table.add_row("[bold pink1]Total bins[/bold pink1]", f"[bold pink1]{total_bins:,}[/bold pink1]")  # no percentage
    table.add_row("[bold]USR bins[/bold]", f"[bold]{fmt(usr_bins, total_bins)}[/bold]")
    table.add_row("[bold]Cell bins[/bold]", f"[bold]{fmt(cell_bins, total_bins)}[/bold]")
    table.add_row("[bold]Overlap bins[/bold]", f"[bold]{fmt(overlap_bins, total_bins)}[/bold]")
    table.add_row("[bold deep_pink2]Axonic bins[/bold deep_pink2]", f"[bold deep_pink2]{fmt(axonic_bins, total_bins)}[/bold deep_pink2]")
    table.add_row("[bold sky_blue1]Dendritic bins[/bold sky_blue1]", f"[bold sky_blue1]{fmt(dendritic_bins, total_bins)}[/bold sky_blue1]")
    table.add_row("[bold orange1]Number of Interfaces[/bold orange1]", f"[bold orange1]{fmt(n_interfaces, total_bins)}[/bold orange1]")
    table.add_row("[bold orange1]Interface bins[/bold orange1]", f"[bold orange1]{fmt(interface_bins, total_bins)}[/bold orange1]")
    table.add_row("[bold bright_green]Significant L-R pairs[/bold bright_green]",f"[bold bright_green]{n_sig} / {n_tested}[/bold bright_green]")
    table.add_row("[bold bright_green]Pairs with null < 0.05[/bold bright_green]",f"[bold bright_green]{n_sig_null} (max={max_null:.2e})[/bold bright_green]")
    histograms = probability_histograms(ad_map, s_map_binary, binary_matrix_ecm)

    table.add_row("[bold deep_pink2]Axonic Probabilities[/bold deep_pink2]", f"[bold deep_pink2]{histograms['Axonic']}[/bold deep_pink2]")
    table.add_row("[bold sky_blue1]Dendritic Probabilities[/bold sky_blue1]", f"[bold sky_blue1]{histograms['Dendritic']}[/bold sky_blue1]")
    table.add_row("[bold orange1]Interface Probabilities[/bold orange1]", f"[bold orange1]{histograms['Interface']}[/bold orange1]")
    panel = Panel(table, style="on grey15", padding=(1, 2), title="Overview", border_style="grey70")
    console.print(panel)

    adata.uns['axon_map'] = a_map
    adata.uns['dendrite_map'] = d_map
    adata.uns['interface_map'] = s_map_binary
    adata.uns['data_shape'] = list(x_shape)
    return adata