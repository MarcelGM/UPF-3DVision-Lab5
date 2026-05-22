import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image


def reproj_err(P, X_h, x_h):
    """Per-point reprojection error of 3D X_h (4xN) in camera P, against 2D x_h (3xN)."""
    proj = P @ X_h
    proj = proj[:2] / proj[2]
    return np.linalg.norm(proj - x_h[:2], axis=0)


# ── Vanishing-point colours (right / left / vertical) ─────────────────────────
_VP_COLOURS = ["#FF6B35", "#00D4FF", "#CC44FF"]
_VP_LABELS  = ["VP 1 — right", "VP 2 — left", "VP 3 — vertical"]


def detect_vps(img_rgb):
    """Detect three orthogonal (Manhattan) vanishing points via LSD + lu-vp-detect.

    Uses the Line Segment Detector (LSD, built into OpenCV) to find subpixel-accurate
    segments, then clusters them into three mutually orthogonal VP directions with
    lu-vp-detect (Xiaohu Lu et al., WACV 2017). Each VP's 2-D image location is
    refined with an accumulator-based intersection voting (same strategy as the
    manual detect_vanishing_points implementation in the notebook).

    Parameters
    ----------
    img_rgb : H×W×3 uint8 RGB array (already resized to working scale)

    Returns
    -------
    vp1, vp2   : (2,) float arrays — the VP pair that gives the largest implied α²
                 (these are the two VPs to use for self-calibration)
    vps_2d     : (3, 2) float array — all three VPs in image-plane pixel coords
    lines      : (N, 4) float array — detected segments [x1, y1, x2, y2]
    clusters   : list of 3 int arrays — line indices belonging to each VP cluster
    """
    import cv2
    from lu_vp_detect import VPDetection
    from scipy.ndimage import gaussian_filter

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    cx, cy  = w / 2.0, h / 2.0

    vpd = VPDetection(length_thresh=15, principal_point=(cx, cy),
                      focal_length=max(w, h), seed=42)
    vpd.find_vps(img_bgr)
    vpd.create_debug_VP_image(show_image=False)

    lines    = vpd._VPDetection__lines        # (N, 4): x1 y1 x2 y2
    clusters = vpd._VPDetection__clusters     # list of 3 index arrays

    def _acc_vp(segs, max_lines=400):
        """Accumulator-based 2-D VP from a line cluster (pairwise intersections)."""
        if len(segs) < 2:
            return None
        lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs = segs[np.argsort(lens)[::-1][:max_lines]].astype(float)
        n    = len(segs)
        p1   = np.c_[segs[:, :2],   np.ones(n)]
        p2   = np.c_[segs[:, 2:4],  np.ones(n)]
        hl   = np.cross(p1, p2)
        CELL, EXT = 16, 20
        AW = int(w * EXT / CELL) + 1
        AH = int(h * EXT / CELL) + 1
        OX, OY = AW // 2, AH // 2
        ii, jj = np.triu_indices(n, k=1)
        pts    = np.cross(hl[ii], hl[jj])
        ok     = np.abs(pts[:, 2]) > 1e-10
        if not ok.any():
            return None
        pts = pts[ok] / pts[ok, 2:3]
        ix  = (pts[:, 0] / CELL).astype(int) + OX
        iy  = (pts[:, 1] / CELL).astype(int) + OY
        inb = (0 <= ix) & (ix < AW) & (0 <= iy) & (iy < AH)
        if not inb.any():
            return None
        acc = np.zeros((AH, AW), dtype=np.float32)
        np.add.at(acc, (iy[inb], ix[inb]), 1.0)
        smooth  = gaussian_filter(acc, sigma=5)
        iy_p, ix_p = np.unravel_index(np.argmax(smooth), smooth.shape)
        return np.array([(ix_p - OX) * CELL, (iy_p - OY) * CELL], dtype=float)

    raw    = [_acc_vp(lines[clusters[k]]) if len(clusters[k]) >= 2 else None
              for k in range(3)]
    vps_2d = np.array([raw[k] if raw[k] is not None else vpd.vps_2D[k]
                       for k in range(3)])

    # Pick the VP pair that maximises α² (IAC formula)
    best_pair, best_a2 = (0, 1), -np.inf
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        xi, yi = vps_2d[i];  xj, yj = vps_2d[j]
        a2 = -(xi - cx) * (xj - cx) - (yi - cy) * (yj - cy)
        if a2 > best_a2:
            best_a2, best_pair = a2, (i, j)

    vp1 = vps_2d[best_pair[0]]
    vp2 = vps_2d[best_pair[1]]
    return vp1, vp2, vps_2d, lines, clusters


def plot_vanishing_points(img_rgb, vps_2d, lines, clusters):
    """Draw three VP panels: lines extended to image boundary + VP marker or arrow.

    Each panel shows the lines belonging to one VP cluster coloured by direction,
    extended to the image boundary, with either a circle+crosshair (VP inside the
    image) or a directional arrow (VP outside the image).

    Parameters
    ----------
    img_rgb  : H×W×3 uint8 array
    vps_2d   : (3, 2) float array — VP image-plane coordinates (from detect_vps)
    lines    : (N, 4) float array — line segments (from detect_vps)
    clusters : list of 3 int arrays — cluster membership (from detect_vps)
    """
    h, w = img_rgb.shape[:2]

    def _clip(x1, y1, x2, y2):
        dx, dy, ts, eps = x2 - x1, y2 - y1, [], 1e-9
        if abs(dx) > eps: ts += [-x1 / dx, (w - x1) / dx]
        if abs(dy) > eps: ts += [-y1 / dy, (h - y1) / dy]
        pts = [(x1 + t * dx, y1 + t * dy) for t in ts
               if -1 <= x1 + t * dx <= w + 1 and -1 <= y1 + t * dy <= h + 1]
        if len(pts) < 2:
            return None
        pts = sorted(pts, key=lambda p: p[0])
        return pts[0], pts[-1]

    def _align(segs, vp):
        mx = (segs[:, 0] + segs[:, 2]) / 2
        my = (segs[:, 1] + segs[:, 3]) / 2
        tv = np.stack([vp[0] - mx, vp[1] - my], axis=1)
        sd = np.stack([segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]], axis=1)
        nv = np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9
        ns = np.linalg.norm(sd, axis=1, keepdims=True) + 1e-9
        return np.abs((tv / nv * sd / ns).sum(axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for k, ax in enumerate(axes):
        vp      = vps_2d[k]
        col     = _VP_COLOURS[k]
        vp_segs = lines[clusters[k]] if len(clusters[k]) else lines[:0]

        ax.imshow(img_rgb, alpha=0.50)

        if len(vp_segs):
            scores = _align(vp_segs, vp)
            keep   = scores >= 0.82
            if keep.sum() < 5:
                keep = scores >= np.percentile(scores, 40)
            sel  = vp_segs[keep]
            lens = np.hypot(sel[:, 2] - sel[:, 0], sel[:, 3] - sel[:, 1])
            for s in sel[np.argsort(lens)[::-1][:35]]:
                clip = _clip(s[0], s[1], s[2], s[3])
                if clip is None:
                    clip = ((s[0], s[1]), (s[2], s[3]))
                (xa, ya), (xb, yb) = clip
                ax.plot([xa, xb], [ya, yb], color=col, lw=1.0,
                        alpha=0.75, solid_capstyle="round")

        inside = (0 <= vp[0] <= w) and (0 <= vp[1] <= h)
        if inside:
            r = min(w, h) * 0.04
            ax.add_patch(plt.Circle(vp, r, color=col, fill=False, lw=2.5, zorder=6))
            ax.plot(vp[0], vp[1], "+", color=col, ms=22, mew=2.5, zorder=7)
            ax.plot(vp[0], vp[1], "o", color="white", ms=5, zorder=8)
        else:
            cxi, cyi = w / 2, h / 2
            dx, dy   = vp[0] - cxi, vp[1] - cyi
            sc = min(w, h) * 0.38 / np.hypot(dx, dy)
            ax.annotate(
                "", xy=(cxi + dx * sc, cyi + dy * sc), xytext=(cxi, cyi),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=2.5, mutation_scale=20), zorder=7)

        loc = "inside" if inside else "outside"
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
        ax.set_title(f"{_VP_LABELS[k]}\n({vp[0]:.0f}, {vp[1]:.0f})  [{loc}]",
                     fontsize=11)

    plt.suptitle(
        "Vanishing Points — LSD + lu-vp-detect (Lu et al., WACV 2017)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def line_draw(line, canv, size):
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]

    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]

    w, h = size

    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    canv.line([beg, end], width=4)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o


def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:, :3]) @ np.array([x[0], x[1], 1])
    return v


def plot_camera(P, w, h, fig, legend, scale=1):

    o = optical_center(P)

    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0] + p2[0]) / 2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1] + p2[1]) / 2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2] + p2[2]) / 2])

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", name=legend))

    return


def plot_camera_col(P, w, h, fig, legend, col, scale=1):

    o = optical_center(P)

    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0] + p2[0]) / 2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1] + p2[1]) / 2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2] + p2[2]) / 2])

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", line=go.scatter3d.Line(color=f"rgb({col})"), name=legend))

    return
