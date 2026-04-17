import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path


def _fmt_count(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} million"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _pct(numerator, denominator):
    if denominator == 0:
        return "0.0%"
    return f"{100 * numerator / denominator:.1f}%"


def _plot_counts(adata):
    mask_ecm = adata.uns["mask_ecm"]
    mask_cell = adata.uns["mask_cell"]
    axon_map = adata.uns["axon_map"]
    dendrite_map = adata.uns["dendrite_map"]
    interface_map = adata.uns["interface_map"]

    return {
        "n_tissue": int(mask_ecm.shape[0] * mask_ecm.shape[1]),
        "n_cell": int(mask_cell.sum()),
        "n_esb": int(mask_ecm.sum()),
        "n_axon": int(axon_map.sum()),
        "n_dendrite": int(dendrite_map.sum()),
        "n_axon_synapse": int((interface_map * axon_map).sum()),
        "n_dendrite_synapse": int((interface_map * dendrite_map).sum()),
    }


def _draw_curve(ax, x1, y1, x2, y2, color="#bbbbbb", lw=1.8):
    verts = [
        (x1, y1),
        (x1, (y1 + y2) / 2),
        (x2, (y1 + y2) / 2),
        (x2, y2),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(
        patches.PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor=color,
            lw=lw,
            zorder=1,
        )
    )


def _draw_node(ax, x, y, label, sublabel, facecolor, textcolor, box_w, box_h):
    ax.add_patch(
        FancyBboxPatch(
            (x - box_w + 0.003, y - box_h - 0.005),
            2 * box_w,
            2 * box_h,
            boxstyle="round,pad=0.012",
            linewidth=0,
            facecolor="#cccccc",
            alpha=0.3,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - box_w, y - box_h),
            2 * box_w,
            2 * box_h,
            boxstyle="round,pad=0.012",
            linewidth=1.5,
            edgecolor="#999999",
            facecolor=to_rgba(facecolor, 0.75),
            zorder=3,
        )
    )
    ax.text(
        x,
        y + box_h * 0.4,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=textcolor,
        zorder=4,
    )
    ax.text(
        x,
        y - box_h * 0.48,
        sublabel,
        ha="center",
        va="center",
        fontsize=9,
        color=textcolor,
        zorder=4,
    )


def tree(adata: ad.AnnData, threshold=None, device="cpu", ax=None, show=True):
    """Plot a vertical hierarchy summary of tissue, ESB, neurites, and interface bins."""
    del threshold, device

    data = _plot_counts(adata)

    box_w = 0.1
    box_h = 0.03

    tissue_x = 0.38
    cell_x = 0.18
    esb_x = 0.55
    axon_x = 0.42
    dendrite_x = 0.68

    interface_w = 0.24
    interface_h = 0.09
    interface_cx = (axon_x + dendrite_x) / 2
    interface_y = 0.325

    red_color = "#ff1a1a"
    blue_color = "#1aa3ff"

    nodes = [
        ("tissue", "Tissue", _fmt_count(data["n_tissue"]), "#ede0ff", "#333333", tissue_x, 0.88),
        ("cell", "Cells", _pct(data["n_cell"], data["n_tissue"]), "#2e2e2e", "#ffffff", cell_x, 0.68),
        ("esb", "Unsegmented", _pct(data["n_esb"], data["n_tissue"]), "#e0e0e0", "#333333", esb_x, 0.68),
        ("axon", "Axon", _pct(data["n_axon"], data["n_esb"]), red_color, "#ffffff", axon_x, 0.5),
        ("dend", "Dendrite", _pct(data["n_dendrite"], data["n_esb"]), blue_color, "#ffffff", dendrite_x, 0.5),
    ]

    edges = [
        ("tissue", "cell"),
        ("tissue", "esb"),
        ("esb", "axon"),
        ("esb", "dend"),
    ]

    n_total_interface = data["n_axon_synapse"] + data["n_dendrite_synapse"]
    n_total_neurite = data["n_axon"] + data["n_dendrite"]

    sx0 = interface_cx - interface_w / 2
    sy0 = interface_y - interface_h / 2
    half_w = interface_w / 2
    node_map = {node[0]: node for node in nodes}

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.figure.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for src_id, tgt_id in edges:
        src = node_map[src_id]
        tgt = node_map[tgt_id]
        _draw_curve(ax, src[5], src[6] - box_h, tgt[5], tgt[6] + box_h)

    _draw_curve(ax, axon_x, node_map["axon"][6] - box_h, sx0 + half_w / 2, interface_y + interface_h / 2)
    _draw_curve(ax, dendrite_x, node_map["dend"][6] - box_h, interface_cx + half_w / 2, interface_y + interface_h / 2)

    for _, label, sublabel, color, textcolor, x, y in nodes:
        _draw_node(ax, x, y, label, sublabel, color, textcolor, box_w, box_h)

    ax.add_patch(
        FancyBboxPatch(
            (sx0 + 0.003, sy0 - 0.005),
            interface_w,
            interface_h,
            boxstyle="round,pad=0.010",
            linewidth=0,
            facecolor="#cccccc",
            alpha=0.3,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (sx0, sy0),
            interface_w,
            interface_h,
            boxstyle="round,pad=0.008",
            linewidth=1.5,
            edgecolor="#f5a800",
            facecolor="#fff8e6",
            zorder=3,
        )
    )

    pad = 0.004
    line_w = 2
    ax.plot([sx0 + pad, sx0 + pad], [sy0 + pad, sy0 + interface_h - pad], color=red_color, lw=line_w, zorder=5)
    ax.plot([sx0 + pad, interface_cx], [sy0 + interface_h - pad, sy0 + interface_h - pad], color=red_color, lw=line_w, zorder=5)
    ax.plot([sx0 + pad, interface_cx], [sy0 + pad, sy0 + pad], color=red_color, lw=line_w, zorder=5)
    ax.plot(
        [sx0 + interface_w - pad, sx0 + interface_w - pad],
        [sy0 + pad, sy0 + interface_h - pad],
        color=blue_color,
        lw=line_w,
        zorder=5,
    )
    ax.plot([interface_cx, sx0 + interface_w - pad], [sy0 + interface_h - pad, sy0 + interface_h - pad], color=blue_color, lw=line_w, zorder=5)
    ax.plot([interface_cx, sx0 + interface_w - pad], [sy0 + pad, sy0 + pad], color=blue_color, lw=line_w, zorder=5)

    label_w = 0.10
    label_h = 0.030
    ax.add_patch(
        plt.Rectangle(
            (interface_cx - label_w / 2, interface_y - label_h / 2),
            label_w,
            label_h,
            facecolor="#fff8e6",
            edgecolor="none",
            zorder=6,
        )
    )
    ax.text(
        interface_cx,
        interface_y + 0.005,
        "Interface",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#222222",
        zorder=7,
    )
    ax.text(
        interface_cx,
        interface_y - 0.013,
        _pct(n_total_interface, n_total_neurite),
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        zorder=7,
    )

    plt.tight_layout()
    if show:
        plt.show()
    return ax
