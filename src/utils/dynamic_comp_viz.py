# =========================



# =========================

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib import font_manager


def set_tcom_style():
    """IEEE TCOM"""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0,
        "grid.color": "0.9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
    })
    
    preferred = ["Times New Roman", "Times", "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Serif"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [name for name in preferred if name in installed]
    if not picked:
        picked = ["DejaVu Serif"]
    plt.rcParams["font.family"] = picked


class DynamicCoMPVisualizer:
    """
    CoMP

    CoMP
    - CoMP
    - //
    - CoMP
    - UAV
    """

    def __init__(
        self,
        trace: Dict[str, Any],
        L: float,
        M: int,
        I: int,
        v: Optional[np.ndarray] = None,
        ris_positions: Optional[np.ndarray] = None,
        enable_ris: bool = True,
        enable_comp: bool = True,
        user_ring_labels: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        trace : dict
            :
            - 'frames': list of dicts with keys 't', 'q', 'a', 'z', 'theta'
            - 'w': user positions (I, 2)
        L : float
            
        M : int
            UAV
        I : int
            
        v : np.ndarray, optional
            RIS (2,)
        enable_ris : bool
            RIS
        enable_comp : bool
            CoMP
        user_ring_labels : List[str], optional
            ["inner", "inner", ..., "mid", ..., "outer"]
        """
        self.trace = trace
        self.L = float(L)
        self.M = int(M)
        self.I = int(I)
        self.ris_positions = self._resolve_ris_positions(v=v, ris_positions=ris_positions)
        self.v = np.asarray(self.ris_positions[0], dtype=np.float64).reshape(2,)
        self.enable_ris = enable_ris
        self.enable_comp = enable_comp
        self.user_ring_labels = user_ring_labels

        self.frames = trace.get("frames", [])
        self.w = np.asarray(trace.get("w", np.zeros((I, 2))), dtype=np.float64).reshape(I, 2)
        self.assoc_threshold = 0.0

        
        if self.user_ring_labels is None:
            self.user_ring_labels = self._infer_user_rings()

    def _resolve_ris_positions(
        self,
        v: Optional[np.ndarray],
        ris_positions: Optional[np.ndarray],
    ) -> np.ndarray:
        if ris_positions is not None:
            try:
                arr = np.asarray(ris_positions, dtype=np.float64)
                if arr.ndim == 1 and arr.size == 2:
                    arr = arr.reshape(1, 2)
                elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                    pass
                else:
                    arr = np.zeros((0, 2), dtype=np.float64)
            except Exception:
                arr = np.zeros((0, 2), dtype=np.float64)
            if arr.size > 0:
                return arr

        try:
            v_arr = np.asarray(v if v is not None else np.array([self.L / 2.0, self.L / 2.0]), dtype=np.float64).reshape(2,)
        except Exception:
            v_arr = np.array([self.L / 2.0, self.L / 2.0], dtype=np.float64)
        return v_arr.reshape(1, 2)

    def _plot_ris(self, ax: Any, show_labels: bool = False) -> None:
        if not self.enable_ris:
            return
        n_ris = int(self.ris_positions.shape[0])
        for idx in range(n_ris):
            pos = self.ris_positions[idx]
            label = "RIS" if idx == 0 else None
            ax.plot(
                pos[0],
                pos[1],
                marker="s",
                markersize=15,
                color="purple",
                markeredgecolor="black",
                markeredgewidth=1.5,
                label=label,
                zorder=10,
            )
            if show_labels and n_ris > 1:
                ax.text(
                    float(pos[0]) + 2.0,
                    float(pos[1]) + 2.0,
                    f"RIS{idx+1}",
                    fontsize=8,
                    fontweight="bold",
                    color="purple",
                    ha="left",
                    va="bottom",
                )

    def _frame_users_xy(self, frame: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        I,2
         trace  `w` trace  `w`
        """
        if isinstance(frame, dict):
            w_raw = frame.get("w", None)
            if w_raw is not None:
                try:
                    return np.asarray(w_raw, dtype=np.float64).reshape(self.I, 2)
                except Exception:
                    pass
        return np.asarray(self.w, dtype=np.float64).reshape(self.I, 2)

    def _infer_user_rings(self) -> List[str]:
        """"""
        center = np.array([self.L/2, self.L/2])
        distances = np.linalg.norm(self.w - center, axis=1)

        
        r1_threshold = 60.0  
        r2_threshold = 115.0  

        labels = []
        for d in distances:
            if d < r1_threshold:
                labels.append("inner")
            elif d < r2_threshold:
                labels.append("mid")
            else:
                labels.append("outer")
        return labels

    def _normalize_assoc_matrix(self, a_raw: Any) -> Optional[np.ndarray]:
        """ trace  (I, M) """
        if a_raw is None:
            return None
        try:
            arr = np.asarray(a_raw, dtype=np.float64)
        except Exception:
            return None
        if arr.size != self.I * self.M:
            return None
        if arr.ndim == 2:
            if arr.shape == (self.I, self.M):
                return np.asarray(arr, dtype=np.float64)
            if arr.shape == (self.M, self.I):
                return np.asarray(arr.T, dtype=np.float64)
        return np.asarray(arr.reshape(self.I, self.M), dtype=np.float64)

    def _build_comp_mask_matrix(self) -> np.ndarray:
        """CoMP (I, T)UAV"""
        T = len(self.frames)
        mask_matrix = np.zeros((self.I, T), dtype=np.int64)

        for t, frame in enumerate(self.frames):
            a = self._normalize_assoc_matrix(frame.get("a", None))
            if a is None:
                continue

            
            threshold = float(self.assoc_threshold)
            for i in range(self.I):
                mask = 0
                for m in range(self.M):
                    if a[i, m] > threshold:
                        mask |= (1 << m)
                mask_matrix[i, t] = mask

        return mask_matrix

    def _build_comp_combo_index(self, mask_matrix: np.ndarray):
        """
        CoMP
        Returns:
            combo_to_id: dict, mask -> combo_id
            id_to_combo: dict, combo_id -> mask
            id_matrix: (I, T), combo_id
            popcount_matrix: (I, T), UAV
        """
        I, T = mask_matrix.shape
        unique_masks = sorted(set(mask_matrix.flatten()))

        combo_to_id = {mask: idx for idx, mask in enumerate(unique_masks)}
        id_to_combo = {idx: mask for mask, idx in combo_to_id.items()}

        id_matrix = np.zeros((I, T), dtype=np.int64)
        popcount_matrix = np.zeros((I, T), dtype=np.int64)

        for i in range(I):
            for t in range(T):
                mask = mask_matrix[i, t]
                id_matrix[i, t] = combo_to_id[mask]
                popcount_matrix[i, t] = bin(mask).count('1')

        return combo_to_id, id_to_combo, id_matrix, popcount_matrix

    def _mask_to_uav_label(self, mask: int, M: int) -> str:
        """UAV"""
        if mask == 0:
            return ""
        uavs = [f"U{m+1}" for m in range(M) if (mask & (1 << m))]
        return "+".join(uavs)

    def _fixed_combo_colors(self, n_combo: int) -> List[str]:
        """CoMP"""
        if n_combo <= 1:
            return ["#E0E0E0"]  

        
        base_colors = plt.cm.tab20.colors
        colors = ["#E0E0E0"]  

        for i in range(1, n_combo):
            colors.append(base_colors[(i-1) % len(base_colors)])

        return colors

    def create_key_frames(self, save_dir: Path, frame_indices: Optional[List[int]] = None):
        """
        UAVCoMP

        
        - 
        - UAV
        - RIS
        - CoMPcomp1comp2
        """
        if frame_indices is None:
            frame_indices = [0, 40, 79]

        set_tcom_style()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        
        comp_colors = [
            "#FF6B6B",  
            "#4ECDC4",  
            "#45B7D1",  
            "#FFA07A",  
            "#98D8C8",  
            "#F7DC6F",  
            "#BB8FCE",  
            "#85C1E2",  
        ]

        def _rect_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
            """"""
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        def _protect_point(
            occupied: List[Tuple[float, float, float, float]],
            ax_obj: Any,
            x: float,
            y: float,
            r_px: float,
        ) -> None:
            """"""
            px, py = ax_obj.transData.transform((x, y))
            occupied.append((float(px - r_px), float(py - r_px), float(px + r_px), float(py + r_px)))

        def _place_label_near(
            ax_obj: Any,
            fig_obj: Any,
            renderer_obj: Any,
            occupied: List[Tuple[float, float, float, float]],
            x: float,
            y: float,
            text: str,
            candidates: List[Tuple[float, float]],
            text_kwargs: Dict[str, Any],
        ) -> None:
            """"""
            for dx, dy in candidates:
                t_obj = ax_obj.text(float(x + dx), float(y + dy), text, **text_kwargs)
                fig_obj.canvas.draw()
                bb = t_obj.get_window_extent(renderer=renderer_obj).expanded(1.08, 1.20)
                rect = (float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1))
                if any(_rect_overlap(rect, occ) for occ in occupied):
                    t_obj.remove()
                    continue
                occupied.append(rect)
                return
            
            dx0, dy0 = candidates[0] if len(candidates) > 0 else (0.0, 0.0)
            t_obj = ax_obj.text(float(x + dx0), float(y + dy0), text, **text_kwargs)
            fig_obj.canvas.draw()
            bb = t_obj.get_window_extent(renderer=renderer_obj).expanded(1.08, 1.20)
            occupied.append((float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)))

        for frame_idx in frame_indices:
            if frame_idx >= len(self.frames):
                continue

            frame = self.frames[frame_idx]
            t = frame.get("t", frame_idx)
            a = frame.get("a", None)
            q = frame.get("q", None)
            w_now = self._frame_users_xy(frame)

            if a is None or q is None:
                continue

            a = self._normalize_assoc_matrix(a)
            if a is None:
                continue
            q = np.asarray(q, dtype=np.float64).reshape(self.M, 2)

            fig, ax = plt.subplots(figsize=(10, 10))
            
            margin = 20  
            ax.set_xlim(-margin, self.L + margin)
            ax.set_ylim(-margin, self.L + margin)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", fontweight="bold")
            ax.set_ylabel("Y (m)", fontweight="bold")
            ax.set_title(f"CoMP @ t={t+1}", fontweight="bold", fontsize=14, pad=15)
            ax.grid(True, alpha=0.3)

            
            self._plot_ris(ax, show_labels=False)

            
            user_color = "#1E88E5"  
            mu_points: List[Tuple[int, float, float]] = []
            for i in range(self.I):
                ax.plot(w_now[i, 0], w_now[i, 1], marker="o", markersize=12,
                       color=user_color, markeredgecolor="black", markeredgewidth=1.0,
                       zorder=5)
                mu_points.append((i + 1, float(w_now[i, 0]), float(w_now[i, 1])))

            
            uav_points: List[Tuple[int, float, float]] = []
            for m in range(self.M):
                ax.plot(q[m, 0], q[m, 1], marker="^", markersize=14,
                       color="#FF5722", markeredgecolor="black", markeredgewidth=1.5,
                       zorder=8)
                uav_points.append((m + 1, float(q[m, 0]), float(q[m, 1])))

            
            threshold = float(self.assoc_threshold)
            user_comp_map = {}  # {user_idx: (serving_uavs_tuple, comp_id)}
            comp_counter = {}   # {serving_uavs_tuple: comp_id}
            next_comp_id = 1

            for i in range(self.I):
                serving_uavs = tuple(sorted([m for m in range(self.M) if a[i, m] > threshold]))
                if len(serving_uavs) >= 1:  
                    if serving_uavs not in comp_counter:
                        comp_counter[serving_uavs] = next_comp_id
                        next_comp_id += 1
                    user_comp_map[i] = (serving_uavs, comp_counter[serving_uavs])

            
            comp_label_specs: List[Tuple[float, float, float, float, str, str]] = []
            for i, (serving_uavs, comp_id) in user_comp_map.items():
                color = comp_colors[(comp_id - 1) % len(comp_colors)]
                for m in serving_uavs:
                    x0, y0 = float(w_now[i, 0]), float(w_now[i, 1])
                    x1, y1 = float(q[m, 0]), float(q[m, 1])
                    ax.plot(
                        [x0, x1], [y0, y1],
                        linestyle="--", linewidth=1.8, color=color, alpha=0.65, zorder=3
                    )
                    
                    xm = 0.55 * x0 + 0.45 * x1
                    ym = 0.55 * y0 + 0.45 * y1
                    vx = x1 - x0
                    vy = y1 - y0
                    norm = float(np.hypot(vx, vy))
                    if norm > 1e-6:
                        
                        nx = -vy / norm
                        ny = vx / norm
                    else:
                        nx, ny = 0.0, 0.0
                    
                    side_offset = 3.6
                    dx = float(nx * side_offset)
                    dy = float(ny * side_offset)
                    comp_label_specs.append((xm, ym, dx, dy, f"comp{comp_id}", color))

            
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            occupied: List[Tuple[float, float, float, float]] = []
            for _, x, y in mu_points:
                _protect_point(occupied, ax, x, y, r_px=11.0)
            for _, x, y in uav_points:
                _protect_point(occupied, ax, x, y, r_px=13.0)
            if self.enable_ris:
                for ridx in range(int(self.ris_positions.shape[0])):
                    rx, ry = float(self.ris_positions[ridx, 0]), float(self.ris_positions[ridx, 1])
                    _protect_point(occupied, ax, rx, ry, r_px=13.0)

            
            for uid, x, y in uav_points:
                _place_label_near(
                    ax, fig, renderer, occupied, x, y, f"UAV{uid}",
                    candidates=[(0.0, 11.0), (8.0, 6.0), (-8.0, 6.0), (0.0, -11.0), (8.0, -6.0), (-8.0, -6.0)],
                    text_kwargs=dict(
                        ha="center", va="center", fontsize=9, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", alpha=0.92, linewidth=1.0),
                        zorder=12,
                    ),
                )

            
            if self.enable_ris and int(self.ris_positions.shape[0]) > 1:
                for ridx in range(int(self.ris_positions.shape[0])):
                    rx, ry = float(self.ris_positions[ridx, 0]), float(self.ris_positions[ridx, 1])
                    _place_label_near(
                        ax, fig, renderer, occupied, rx, ry, f"RIS{ridx+1}",
                        candidates=[(3.5, 3.0), (-3.5, 3.0), (3.5, -3.0), (-3.5, -3.0), (0.0, 5.5), (0.0, -5.5)],
                        text_kwargs=dict(
                            ha="center", va="center", fontsize=8, fontweight="bold", color="purple",
                            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="purple", alpha=0.90, linewidth=0.9),
                            zorder=12,
                        ),
                    )

            
            for uid, x, y in mu_points:
                _place_label_near(
                    ax, fig, renderer, occupied, x, y, f"MU{uid}",
                    candidates=[(0.0, -6.0), (4.5, -4.0), (-4.5, -4.0), (0.0, 6.0), (4.5, 3.5), (-4.5, 3.5)],
                    text_kwargs=dict(
                        ha="center", va="center", fontsize=8, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.78),
                        zorder=12,
                    ),
                )

            
            for xm, ym, bdx, bdy, txt_label, color in comp_label_specs:
                cand = [
                    (bdx, bdy),
                    (-bdx, -bdy),
                    (0.75 * bdx + 1.2, 0.75 * bdy - 0.8),
                    (-0.75 * bdx - 1.2, -0.75 * bdy + 0.8),
                    (1.15 * bdx, 1.15 * bdy),
                ]
                _place_label_near(
                    ax, fig, renderer, occupied, xm, ym, txt_label,
                    candidates=[(float(cx), float(cy)) for cx, cy in cand],
                    text_kwargs=dict(
                        ha="center", va="center", fontsize=6.2, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=color, alpha=0.92, linewidth=1.0),
                        zorder=11,
                    ),
                )

            
            from matplotlib.patches import Patch
            legend_elements = [
                plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=user_color, markeredgecolor="black",
                          markersize=10, label="(MU)"),
                plt.Line2D([0], [0], marker="^", color="w",
                          markerfacecolor="#FF5722", markeredgecolor="black",
                          markersize=10, label="(UAV)"),
            ]
            if self.enable_ris:
                n_ris = int(self.ris_positions.shape[0])
                ris_label = "RIS" if n_ris <= 1 else f"RIS(x{n_ris})"
                legend_elements.append(plt.Line2D([0], [0], marker="s", color="w",
                                                  markerfacecolor="purple", markersize=10, label=ris_label))

            
            shown_comps = sorted(comp_counter.items(), key=lambda x: x[1])
            for serving_uavs, comp_id in shown_comps:
                color = comp_colors[(comp_id - 1) % len(comp_colors)]
                uav_label = "+".join([f"UAV{m+1}" for m in serving_uavs])
                legend_elements.append(plt.Line2D([0], [0], linestyle="--", linewidth=2.0,
                                                  color=color, label=f"comp{comp_id}: {uav_label}"))

            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

            save_path = save_dir / f"key_frame_t{t+1:04d}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved key frame: {save_path}")

    def create_comp_timeline(self, save_path: Path, max_combo_legend: int = 18):
        """
        CoMPcomp_timeline.png

        
        - IDcomp1/comp2
        - 
        - CoMP|S|>=2
        """
        set_tcom_style()

        if not self.frames:
            print("[WARN] No frames in trace, skipping CoMP timeline")
            return

        mask_matrix = self._build_comp_mask_matrix()  # (I, T)
        I, T = mask_matrix.shape
        combo_to_id, id_to_combo, id_matrix, popcount_matrix = self._build_comp_combo_index(mask_matrix)
        n_combo = len(combo_to_id)

        
        if T >= 2:
            switch_counts = np.sum(mask_matrix[:, 1:] != mask_matrix[:, :-1], axis=1).astype(np.int64)
        else:
            switch_counts = np.zeros((I,), dtype=np.int64)

        
        comp_ratio = np.mean(popcount_matrix >= 2, axis=1)

        
        fig_w = float(max(14.0, min(30.0, 8.5 + 0.22 * T)))
        fig_h = float(max(8.0, min(14.0, 7.2 + 0.20 * I)))
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[3.4, 1.2],
            width_ratios=[3.0, 3.0, 2.7],
            hspace=0.30,
            wspace=0.24,
        )

        ax_timeline = fig.add_subplot(gs[0, :2])
        ax_legend = fig.add_subplot(gs[0, 2])
        ax_switch = fig.add_subplot(gs[1, :2])
        ax_ratio = fig.add_subplot(gs[1, 2])

        
        palette = self._fixed_combo_colors(n_combo)
        cmap = mcolors.ListedColormap(palette)
        boundaries = np.arange(-0.5, float(n_combo) + 0.5, 1.0)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=n_combo)

        
        im = ax_timeline.imshow(
            id_matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        ax_timeline.set_xlabel(" t", fontweight="bold")
        ax_timeline.set_ylabel("", fontweight="bold")
        ax_timeline.set_yticks(np.arange(I))
        ax_timeline.set_yticklabels([f"MU{i + 1}" for i in range(I)])
        if T <= 36:
            ax_timeline.set_xticks(np.arange(T))
        else:
            tick_num = min(12, T)
            ax_timeline.set_xticks(np.linspace(0, T - 1, num=tick_num))

        
        if I * T <= 300:
            for i in range(I):
                for t in range(T):
                    cid = int(id_matrix[i, t])
                    txt = "none" if cid == 0 else f"comp{cid}"
                    ax_timeline.text(
                        t, i, txt,
                        ha="center", va="center",
                        fontsize=6.5, color="black", alpha=0.90,
                    )

        ax_timeline.set_title("CoMP", fontweight="bold", fontsize=14)

        
        ax_legend.set_xlim(0.0, 1.0)
        ax_legend.set_ylim(0.0, 1.0)
        ax_legend.axis("off")
        ax_legend.set_title("", fontweight="bold", fontsize=14, pad=9)

        legend_entries: List[Tuple[int, str, str]] = [(0, "none", "/")]
        show_n = int(max(0, min(max_combo_legend, n_combo - 1)))
        for cid in range(1, show_n + 1):
            mask = int(id_to_combo[cid])
            legend_entries.append((cid, f"comp{cid}", self._mask_to_uav_label(mask, self.M)))
        remain = int((n_combo - 1) - show_n)

        n_entries = len(legend_entries)
        n_cols = 1 if n_entries <= 12 else 2
        rows_per_col = int(np.ceil(n_entries / float(n_cols)))
        usable_h = 0.90
        row_h = usable_h / max(rows_per_col, 1)

        for idx, (cid, comp_label, uav_label) in enumerate(legend_entries):
            col = idx // rows_per_col
            row = idx % rows_per_col
            x_base = 0.05 + col * 0.50
            y_pos = 0.95 - (row + 0.5) * row_h

            color = palette[cid]
            rect = FancyBboxPatch(
                (x_base, y_pos - 0.015), 0.08, 0.030,
                boxstyle="round,pad=0.003",
                facecolor=color, edgecolor="black", linewidth=0.8,
                transform=ax_legend.transAxes, zorder=2
            )
            ax_legend.add_patch(rect)

            ax_legend.text(
                x_base + 0.10, y_pos,
                f"{comp_label} = {uav_label}",
                ha="left", va="center",
                fontsize=8, transform=ax_legend.transAxes
            )

        if remain > 0:
            ax_legend.text(
                0.5, 0.02,
                f" {remain} ",
                ha="center", va="bottom",
                fontsize=7, style="italic", color="gray",
                transform=ax_legend.transAxes
            )

        
        ax_switch.bar(np.arange(I), switch_counts, color="steelblue", edgecolor="black", linewidth=0.8)
        ax_switch.set_xlabel("", fontweight="bold")
        ax_switch.set_ylabel("", fontweight="bold")
        ax_switch.set_title("", fontweight="bold", fontsize=12)
        ax_switch.set_xticks(np.arange(I))
        ax_switch.set_xticklabels([f"{i+1}" for i in range(I)])
        ax_switch.grid(axis="y", alpha=0.3)

        
        ax_ratio.bar(np.arange(I), comp_ratio * 100, color="steelblue", edgecolor="black", linewidth=0.8)
        ax_ratio.set_xlabel("", fontweight="bold")
        ax_ratio.set_ylabel("(%)", fontweight="bold")
        ax_ratio.set_title("CoMP(|S|>=2)", fontweight="bold", fontsize=12)
        ax_ratio.set_xticks(np.arange(I))
        ax_ratio.set_xticklabels([f"{i+1}" for i in range(I)])
        ax_ratio.set_ylim(0, 100)
        ax_ratio.grid(axis="y", alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved CoMP timeline: {save_path}")

    def _get_ring_color(self, ring: str) -> str:
        """"""
        ring_colors = {"inner": "#FF6B6B", "mid": "#4ECDC4", "outer": "#45B7D1"}
        return ring_colors.get(ring, "gray")

    def create_user_focus_animation(
        self,
        save_path: Path,
        target_user_idx: int = 0,
        fps: int = 5,
        dpi: int = 100,
    ):
        """
        CoMPuser_focus_animation.gif

        
        - CoMP
        - 
        - UAV
        """
        set_tcom_style()

        if not self.frames:
            print("[WARN] No frames in trace, skipping user focus animation")
            return

        T = len(self.frames)
        target_user_idx = int(target_user_idx)
        if target_user_idx < 0 or target_user_idx >= self.I:
            print(f"[WARN] Invalid target_user_idx={target_user_idx}, using 0")
            target_user_idx = 0

        
        mask_matrix = self._build_comp_mask_matrix()
        combo_to_id, id_to_combo, id_matrix, popcount_matrix = self._build_comp_combo_index(mask_matrix)
        n_combo = len(combo_to_id)
        palette = self._fixed_combo_colors(n_combo)

        
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
        ax_spatial = fig.add_subplot(gs[0])
        ax_timeline = fig.add_subplot(gs[1])

        
        ax_spatial.set_xlim(0, self.L)
        ax_spatial.set_ylim(0, self.L)
        ax_spatial.set_aspect("equal")
        ax_spatial.set_xlabel("X (m)", fontweight="bold")
        ax_spatial.set_ylabel("Y (m)", fontweight="bold")
        ax_spatial.grid(True, alpha=0.3)

        
        self._plot_ris(ax_spatial, show_labels=False)

        
        user_color = "#1E88E5"  
        w0 = self._frame_users_xy(self.frames[0] if self.frames else None)
        other_idx = [i for i in range(self.I) if i != target_user_idx]
        other_users_scatter = None
        if len(other_idx) > 0:
            other_users_scatter = ax_spatial.scatter(
                w0[other_idx, 0],
                w0[other_idx, 1],
                s=70,
                c=user_color,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.5,
                zorder=5,
            )
        target_user_marker, = ax_spatial.plot(
            [w0[target_user_idx, 0]],
            [w0[target_user_idx, 1]],
            marker="o",
            markersize=18,
            color="gold",
            markeredgecolor="red",
            markeredgewidth=2.5,
            linestyle="None",
            zorder=8,
        )
        target_user_text = ax_spatial.text(
            w0[target_user_idx, 0],
            w0[target_user_idx, 1] + 10,
            f"MU{target_user_idx+1}*",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="red",
        )

        
        ax_timeline.set_xlim(0, T)
        ax_timeline.set_ylim(-0.5, 0.5)
        ax_timeline.set_xlabel(" t", fontweight="bold")
        ax_timeline.set_title(f"MU{target_user_idx+1} ", fontweight="bold", fontsize=12)
        ax_timeline.set_yticks([])
        ax_timeline.grid(axis="x", alpha=0.3)

        
        for t in range(T):
            cid = int(id_matrix[target_user_idx, t])
            color = palette[cid]
            ax_timeline.add_patch(Rectangle((t, -0.4), 1, 0.8, facecolor=color, edgecolor="black", linewidth=0.5))

        
        comp_lines = []
        time_marker = ax_timeline.axvline(0, color="red", linewidth=2, zorder=10)

        def update(frame_idx):
            
            for line in comp_lines:
                line.remove()
            comp_lines.clear()

            
            frame = self.frames[frame_idx]
            t = frame.get("t", frame_idx)
            a = frame.get("a", None)
            q = frame.get("q", None)
            w_now = self._frame_users_xy(frame)

            
            if other_users_scatter is not None and len(other_idx) > 0:
                other_users_scatter.set_offsets(w_now[other_idx, :2])
            target_user_marker.set_data([w_now[target_user_idx, 0]], [w_now[target_user_idx, 1]])
            target_user_text.set_position((w_now[target_user_idx, 0], w_now[target_user_idx, 1] + 10))

            if a is not None and q is not None:
                a = self._normalize_assoc_matrix(a)
                if a is None:
                    artists = comp_lines + [time_marker, target_user_marker, target_user_text]
                    if other_users_scatter is not None:
                        artists.append(other_users_scatter)
                    return artists
                q = np.asarray(q, dtype=np.float64).reshape(self.M, 2)

                
                threshold = float(self.assoc_threshold)
                serving_uavs = [m for m in range(self.M) if a[target_user_idx, m] > threshold]
                for m in serving_uavs:
                    line, = ax_spatial.plot(
                        [w_now[target_user_idx, 0], q[m, 0]],
                        [w_now[target_user_idx, 1], q[m, 1]],
                        linestyle="--", linewidth=2.0, color="orange", alpha=0.8, zorder=7
                    )
                    comp_lines.append(line)

                    
                    text = ax_spatial.text(
                        q[m, 0], q[m, 1] + 8, f"UAV{m+1}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
                    )
                    comp_lines.append(text)

            
            time_marker.set_xdata([frame_idx])

            
            cid = int(id_matrix[target_user_idx, frame_idx])
            mask = int(id_to_combo[cid])
            comp_label = "none" if cid == 0 else f"comp{cid}"
            uav_label = self._mask_to_uav_label(mask, self.M)
            ax_spatial.set_title(
                f" MU{target_user_idx+1} @ t={t}\n: {comp_label} ({uav_label})",
                fontweight="bold", fontsize=14
            )

            artists = comp_lines + [time_marker, target_user_marker, target_user_text]
            if other_users_scatter is not None:
                artists.append(other_users_scatter)
            return artists

        
        anim = FuncAnimation(fig, update, frames=range(T), interval=1000//fps, blit=False, repeat=True)

        
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Saved user focus animation: {save_path}")

    def export_comp_metrics_csv(self, save_path: Path):
        """CoMPCSV"""
        if not self.frames:
            print("[WARN] No frames in trace, skipping CSV export")
            return

        mask_matrix = self._build_comp_mask_matrix()
        I, T = mask_matrix.shape
        combo_to_id, id_to_combo, id_matrix, popcount_matrix = self._build_comp_combo_index(mask_matrix)

        
        if T >= 2:
            switch_counts = np.sum(mask_matrix[:, 1:] != mask_matrix[:, :-1], axis=1).astype(np.int64)
        else:
            switch_counts = np.zeros((I,), dtype=np.int64)

        comp_ratio = np.mean(popcount_matrix >= 2, axis=1)
        avg_serving_uavs = np.mean(popcount_matrix, axis=1)

        
        ring_stats = {}
        for ring in ["inner", "mid", "outer"]:
            indices = [i for i in range(I) if self.user_ring_labels[i] == ring]
            if indices:
                ring_stats[ring] = {
                    "count": len(indices),
                    "avg_switch": float(np.mean(switch_counts[indices])),
                    "avg_comp_ratio": float(np.mean(comp_ratio[indices])),
                    "avg_serving_uavs": float(np.mean(avg_serving_uavs[indices])),
                }

        
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Users", I])
            writer.writerow(["Total Timesteps", T])
            writer.writerow(["Total CoMP Combos", len(combo_to_id)])
            writer.writerow([])

            writer.writerow(["User", "Ring", "Switch Count", "CoMP Ratio", "Avg Serving UAVs"])
            for i in range(I):
                writer.writerow([
                    f"MU{i+1}",
                    self.user_ring_labels[i],
                    int(switch_counts[i]),
                    f"{comp_ratio[i]:.4f}",
                    f"{avg_serving_uavs[i]:.4f}",
                ])

            writer.writerow([])
            writer.writerow(["Ring Statistics"])
            writer.writerow(["Ring", "User Count", "Avg Switch", "Avg CoMP Ratio", "Avg Serving UAVs"])
            for ring in ["inner", "mid", "outer"]:
                if ring in ring_stats:
                    stats = ring_stats[ring]
                    writer.writerow([
                        ring,
                        stats["count"],
                        f"{stats['avg_switch']:.2f}",
                        f"{stats['avg_comp_ratio']:.4f}",
                        f"{stats['avg_serving_uavs']:.4f}",
                    ])

        print(f"[INFO] Saved CoMP metrics CSV: {save_path}")

    def generate_all_visualizations(
        self,
        output_dir: Path,
        target_user_idx: int = 0,
        key_frame_indices: Optional[List[int]] = None,
    ):
        """
        CoMP

        Parameters
        ----------
        output_dir : Path
            
        target_user_idx : int
            
        key_frame_indices : List[int]
            
        """
        if key_frame_indices is None:
            key_frame_indices = [0, 40, 79]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Generating dynamic CoMP visualizations...")

        
        key_frames_dir = output_dir / "key_frames"
        self.create_key_frames(key_frames_dir, key_frame_indices)

        
        timeline_path = output_dir / "comp_timeline.png"
        self.create_comp_timeline(timeline_path)

        
        animation_path = output_dir / "user_focus_animation.gif"
        self.create_user_focus_animation(animation_path, target_user_idx)

        
        csv_path = output_dir / "comp_metrics.csv"
        self.export_comp_metrics_csv(csv_path)

        print(f"[INFO] All visualizations saved to: {output_dir}")


def visualize_dynamic_comp(
    trace: Dict[str, Any],
    output_dir: Path,
    L: float,
    M: int,
    I: int,
    v: Optional[np.ndarray] = None,
    ris_positions: Optional[np.ndarray] = None,
    enable_ris: bool = True,
    enable_comp: bool = True,
    user_ring_labels: Optional[List[str]] = None,
    target_user_idx: int = 0,
    key_frame_indices: Optional[List[int]] = None,
):
    """
    CoMP

    Parameters
    ----------
    trace : dict
        
    output_dir : Path
        
    L : float
        
    M : int
        UAV
    I : int
        
    v : np.ndarray, optional
        RIS
    enable_ris : bool
        RIS
    enable_comp : bool
        CoMP
    user_ring_labels : List[str], optional
        
    target_user_idx : int
        
    key_frame_indices : List[int]
        
    """
    viz = DynamicCoMPVisualizer(
        trace=trace,
        L=L,
        M=M,
        I=I,
        v=v,
        ris_positions=ris_positions,
        enable_ris=enable_ris,
        enable_comp=enable_comp,
        user_ring_labels=user_ring_labels,
    )

    viz.generate_all_visualizations(
        output_dir=output_dir,
        target_user_idx=target_user_idx,
        key_frame_indices=key_frame_indices,
    )

