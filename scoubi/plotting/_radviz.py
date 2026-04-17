from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from adjustText import adjust_text

def radviz(df_profile, cell_types = None, group_mapping = None, fraction_threshold = 0.1,
                     bandwidth=1, max_alpha=0.65, grid_size=400, beta = 10,
                     jitter_strength=0.05, eps=0.02, pad=1.1, s_scale=100,
                     radius_threshold=0.25,
                     class_colors_hex=None, annotate = False, annotation_radius = 0.75, annotation_size = None, return_genes = False, ax = None, show = True, annotation_fontsize = 9):
    """
    Generate a Radviz plot with KDE-based background for groups.
    Points near center are assigned 'None' group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original feature dataframe (samples x features)
    group_mapping : dict
        Mapping of group_name -> list of features
    bandwidth : float
        KDE bandwidth (larger = smoother background)
    max_alpha : float
        Max opacity for densest KDE region
    grid_size : int
        Resolution of background grid
    jitter_strength : float
        Maximum jitter applied to overlapping points
    eps : float
        Radius threshold to consider points overlapping
    pad : float
        Grid extent padding
    radius_threshold : float
        Radius for 'None' group assignment (near center)
    class_colors_hex : list of str
        Hex color codes for each group, e.g. ['#FF6666', '#3399FF', '#33CC33']
    """
    df = df_profile.copy()
    # ---------------- Default Colors ----------------
    if cell_types is not None:
        df.drop(columns=[c for c in df.columns if c not in cell_types], inplace=True)
        df = df[cell_types]
    if group_mapping is None:
        group_mapping = {col: [col] for col in df.columns}
    n_groups = len(group_mapping)
    if class_colors_hex is None:
        cmap = plt.get_cmap("rainbow", n_groups)
        class_colors_hex = [mcolors.to_hex(cmap(i)) for i in range(n_groups)]
    class_colors = np.array([mcolors.to_rgb(h) for h in class_colors_hex])
    group_to_color = {g: class_colors[i] for i, g in enumerate(list(group_mapping.keys()))}
    # ---------------- Step 1: Prepare Data ----------------
    temp = df.copy()
    temp[temp<0] = 0
    temp = temp.loc[temp.sum(axis=1) > 0]
    ordered_cols = [f for g in group_mapping for f in group_mapping[g] if f in temp.columns]
    df_reordered = temp[ordered_cols]
    # df_reordered = df_reordered.loc[df_reordered.sum(axis=1) > 0]

    # ---------------- Step 2: Compute Radviz Coordinates ----------------
    def compute_radviz(data, beta=beta):
        cols = data.columns
        n = len(cols)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        S = pd.DataFrame(np.c_[np.cos(angles), np.sin(angles)], index=cols, columns=["x", "y"])
        df_norm = (data[cols] - data[cols].min()) / (data[cols].max() - data[cols].min())
        df_softmax = np.exp(beta * df_norm)
        df_softmax = df_softmax.div(df_softmax.sum(axis=1), axis=0)
        coords_xy = df_softmax.dot(S)
        r = np.sqrt((coords_xy**2).sum(axis=1)).max()
        coords_xy = coords_xy / r
        gene_strengths = data.max(axis=1).clip(0, 1).values
        coords = pd.DataFrame(coords_xy, columns=["x", "y"], index=data.index)
        coords["s"] = gene_strengths
        return df_norm, coords, S

    gene_strengths, coords, anchors = compute_radviz(df_reordered)
    #remove genes with max strength < fraction_threshold
    mask = gene_strengths.max(axis=1) >= fraction_threshold
    gene_strengths = gene_strengths[mask]
    coords = coords[mask]
    
    # ---------------- Step 3: Jitter overlapping points ----------------
    tree = KDTree(coords[['x','y']].values)
    neighbors = tree.query_radius(coords[['x','y']].values, r=eps)
    x_jitter = coords['x'].values.copy()
    y_jitter = coords['y'].values.copy()
    for i, inds in enumerate(neighbors):
        if len(inds) > 1:
            # random jitter in x and y
            dx, dy = np.random.normal(0, jitter_strength, 2)
            x_new = x_jitter[i] + dx
            y_new = y_jitter[i] + dy
            
            # check distance from center
            r_new = np.sqrt(x_new**2 + y_new**2)
            if r_new > 1.0:
                # scale it back to lie just inside the circle
                scale = 0.99 / r_new
                x_new *= scale
                y_new *= scale
            
            x_jitter[i] = x_new
            y_jitter[i] = y_new

    coords['x_jitter'] = x_jitter
    coords['y_jitter'] = y_jitter

    # ---------------- Step 4: Assign nearest group ----------------
    anchor_to_group = {}
    for group, features in group_mapping.items():
        for f in features:
            if f in anchors.index:
                anchor_to_group[f] = group

    # Distances to anchors
    anchor_positions = anchors[['x','y']].values
    dists_to_anchors = pairwise_distances(coords[['x','y']].values, anchor_positions)
    nearest_anchor_idx = dists_to_anchors.argmin(axis=1)
    nearest_anchors = anchors.index[nearest_anchor_idx]

    vals = gene_strengths.values
    max_idx = np.argmax(vals, axis=1)
    sorted_vals = np.sort(vals, axis=1)
    max_strength = sorted_vals[:, -1]
    second_strength = sorted_vals[:, -2]
    use_strength = (max_strength - second_strength) > (0.5 * max_strength)

    coords['assigned_group'] = np.where(
        use_strength,
        gene_strengths.columns[max_idx],
        nearest_anchors
    )

    coords['radius'] = np.linalg.norm(coords[['x','y']].values, axis=1)
    coords['assigned_group'] = np.where(coords['radius'] <= radius_threshold, 'None', coords['assigned_group'])
    unique_groups = np.unique(coords['assigned_group'])

    # ---------------- Step 5: KDE Background ----------------
    xx, yy = np.meshgrid(np.linspace(-pad, pad, grid_size), np.linspace(-pad, pad, grid_size))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    inside_mask = (xx**2 + yy**2) <= 1.0
    inside_mask_flat = inside_mask.ravel()

    alpha_stack = np.zeros((grid_pts.shape[0], len(unique_groups)))
    colors_for_kde = []

    for k, g in enumerate(unique_groups):
        pts = coords.loc[coords['assigned_group']==g, ['x_jitter','y_jitter']].values
        # print(pts)
        # if pts.shape[0] >= 2:
        #     kde = gaussian_kde(pts.T, bw_method=bandwidth)
        #     dens = kde(grid_pts.T)
        # elif pts.shape[0] == 1:
        #     mean = pts[0]
        #     sigma = 0.08
        #     d2 = np.sum((grid_pts - mean[None,:])**2, axis=1)
        #     dens = np.exp(-0.5*d2/sigma**2)
        # else:
        dens = np.zeros(grid_pts.shape[0])
        dens[~inside_mask_flat] = 0           # keep outside circle white
        dens = np.clip(dens, 0, None)         # remove negatives
        dens_norm = dens / dens.max() if dens.max() > 0 else dens
        dens_norm = np.clip(dens_norm, 0, 1)   # important!
        alpha_stack[:, k] = dens_norm * max_alpha
        alpha_stack[:, k] = np.minimum(dens_norm * max_alpha, max_alpha)

        # assign color for each group
        if g == 'None':
            colors_for_kde.append(np.array([1.0,1.0,1.0]))
        else:
            # idx = list(unique_groups).index(g) % len(class_colors)
            colors_for_kde.append(group_to_color[g])
    colors_for_kde = np.array(colors_for_kde)

    # Composite colors over white
    alpha_sum = np.clip(alpha_stack.sum(axis=1), 0, 1)
    color_contrib = alpha_stack.dot(colors_for_kde)
    final_pixels = color_contrib + (1 - alpha_sum)[:, None]
    final_pixels = np.clip(final_pixels, 0, 1)
    final_pixels[~inside_mask_flat, :] = 1.0
    final_image = final_pixels.reshape(grid_size, grid_size, 3)

    # ---------------- Step 6: Plot ----------------
    if ax is None:
        fig, ax_ = plt.subplots(figsize=(10,10))
    else:
        ax_ = ax
    ax_.imshow(final_image, origin='lower', extent=(-pad,pad,-pad,pad), zorder=0)

    # Anchors
    for f in anchors.index:
        ax_.scatter(anchors.loc[f,'x'], anchors.loc[f,'y'], s=30, color='0.3', edgecolor='black', linewidth=0.6, alpha=0.8, zorder=4)
        ax_.text(1.15*anchors.loc[f,'x'], 1.15*anchors.loc[f,'y'], f,
                ha='center', va='center', fontsize=12, weight='bold', zorder=5)
    
    # Fade-to-white for points
    fade = 1 - np.clip(coords['radius'].values, 0, 1)
    blended_colors = []
    alphas = []
    for i, g in enumerate(coords['assigned_group']):
        if g=='None':
            col = np.array([0.85,0.85,0.85])
            a = 0.3
        else:
            # idx = list(unique_groups).index(g) % len(class_colors)
            col = group_to_color[g]
            a = np.clip(1*coords.iloc[i]['s'], 0.8, 1)
        final_col = (1-fade[i])*col + fade[i]*np.array([1,1,1])
        blended_colors.append(final_col)
        alphas.append(a)
    ax_.scatter(coords['x_jitter'], coords['y_jitter'],
               facecolors=blended_colors, edgecolors='black',
               linewidths=0.4, s=s_scale * coords.s.values, alpha=alphas, zorder=3)
    if annotate:
        texts = []
        for i, row in coords.iterrows():
            # if row['assigned_group'] == 'None':
            #     continue
            if row['radius'] < annotation_radius:
                continue
            if annotation_size is not None and row['s'] < annotation_size:
                continue

            texts.append(
                ax_.text(
                    row['x_jitter'], row['y_jitter'], i,
                    fontsize=annotation_fontsize, fontweight='bold',
                    color='black', zorder=4
                )
            )
        adjust_text(texts, ax=ax_, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        coords['assigned_group'] = np.where(coords['radius'].round(2) <= annotation_radius, 'None', coords['assigned_group'])
    # Circle boundary
    circle = plt.Circle((0,0),1.0,color='gray',fill=False,linestyle='--',linewidth=1.2,zorder=2)
    ax_.add_artist(circle)

    # ax_.set_clip_on(False)
    # ax_.set_xlim(-2.5,2.5)
    # ax_.set_ylim(-2.5,2.5)

    ax_.set_aspect('equal')
    ax_.axis('off')
    # plt.tight_layout()
    if show:
        plt.show()

    if ax is not None:
        if return_genes:
            return ax_, coords
        return ax_
    plt.close()
    if return_genes:
        return coords