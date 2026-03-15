# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Residual-stream analysis: geometry statistics and PaCMAP projection plots."""

from pathlib import Path

import torch
import torch.linalg as LA
import torch.nn.functional as F
from rich.progress import track
from rich.table import Table
from torch import Tensor

from .settings import PrometheusConfig
from .util import print


class ResidualAnalyzer:
    """Computes and visualises per-layer residual-stream geometry.

    Instantiated once with the benign / target residual tensors produced
    during steering-vector computation.  The heavy-lifting (geometric median,
    PaCMAP, silhouette) requires the ``[research]`` optional dependency group.
    """

    def __init__(
        self,
        config: PrometheusConfig,
        engine,
        benign_states: Tensor,
        target_states: Tensor,
    ):
        self.config = config
        self.engine = engine
        self.benign_states = benign_states
        self.target_states = target_states

    # ------------------------------------------------------------------
    # Tabular geometry report
    # ------------------------------------------------------------------

    def print_residual_geometry(self):
        try:
            from geom_median.torch import (  # ty:ignore[unresolved-import]
                compute_geometric_median,
            )
            from sklearn.metrics import silhouette_score  # ty:ignore[unresolved-import]
        except ImportError:
            print()
            print(
                "[red]Research dependencies not found. Printing residual geometry requires "
                "installing Prometheus with the optional research feature, i.e., "
                'using "pip install -U prometheus-llm\\[research]".[/]'
            )
            return

        print()
        print("Computing residual geometry...")

        table = Table()
        for col in [
            "Layer",
            "S(g,b)",
            "S(g*,b*)",
            "S(g,r)",
            "S(g*,r*)",
            "S(b,r)",
            "S(b*,r*)",
            "|g|",
            "|g*|",
            "|b|",
            "|b*|",
            "|r|",
            "|r*|",
            "Silh",
        ]:
            table.add_column(col, justify="right")

        n_layers = len(self.engine.transformer_layers)

        g = self.benign_states.mean(dim=0)
        g_star = torch.stack(
            [
                compute_geometric_median(
                    self.benign_states[:, li, :].detach().cpu(),
                ).median
                for li in range(n_layers + 1)
            ]
        )
        b = self.target_states.mean(dim=0)
        b_star = torch.stack(
            [
                compute_geometric_median(
                    self.target_states[:, li, :].detach().cpu(),
                ).median
                for li in range(n_layers + 1)
            ]
        )
        r = b - g
        r_star = b_star - g_star

        sim_gb = F.cosine_similarity(g, b, dim=-1)
        sim_gsbs = F.cosine_similarity(g_star, b_star, dim=-1)
        sim_gr = F.cosine_similarity(g, r, dim=-1)
        sim_gsrs = F.cosine_similarity(g_star, r_star, dim=-1)
        sim_br = F.cosine_similarity(b, r, dim=-1)
        sim_bsrs = F.cosine_similarity(b_star, r_star, dim=-1)

        norm_g = LA.vector_norm(g, dim=-1)
        norm_gs = LA.vector_norm(g_star, dim=-1)
        norm_b = LA.vector_norm(b, dim=-1)
        norm_bs = LA.vector_norm(b_star, dim=-1)
        norm_r = LA.vector_norm(r, dim=-1)
        norm_rs = LA.vector_norm(r_star, dim=-1)

        combined = (
            torch.cat([self.benign_states, self.target_states], dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        labels = [0] * len(self.benign_states) + [1] * len(self.target_states)
        silhouettes = [
            silhouette_score(combined[:, li, :], labels) for li in range(n_layers + 1)
        ]

        for li in range(1, n_layers + 1):
            table.add_row(
                f"{li}",
                f"{sim_gb[li].item():.4f}",
                f"{sim_gsbs[li].item():.4f}",
                f"{sim_gr[li].item():.4f}",
                f"{sim_gsrs[li].item():.4f}",
                f"{sim_br[li].item():.4f}",
                f"{sim_bsrs[li].item():.4f}",
                f"{norm_g[li].item():.2f}",
                f"{norm_gs[li].item():.2f}",
                f"{norm_b[li].item():.2f}",
                f"{norm_bs[li].item():.2f}",
                f"{norm_r[li].item():.2f}",
                f"{norm_rs[li].item():.2f}",
                f"{silhouettes[li]:.4f}",
            )

        print()
        print("[bold]Residual Geometry[/]")
        print(table)
        print("[bold]g[/] = mean of residual vectors for benign prompts")
        print("[bold]g*[/] = geometric median of residual vectors for benign prompts")
        print("[bold]b[/] = mean of residual vectors for target prompts")
        print("[bold]b*[/] = geometric median of residual vectors for target prompts")
        print("[bold]r[/] = steering direction for means (i.e., [bold]b - g[/])")
        print(
            "[bold]r*[/] = steering direction for geometric medians (i.e., [bold]b* - g*[/])"
        )
        print("[bold]S(x,y)[/] = cosine similarity of [bold]x[/] and [bold]y[/]")
        print("[bold]|x|[/] = L2 norm of [bold]x[/]")
        print(
            "[bold]Silh[/] = Mean silhouette coefficient of residuals for benign/target clusters"
        )

    # ------------------------------------------------------------------
    # PaCMAP projection animation
    # ------------------------------------------------------------------

    def plot_residuals(self):
        try:
            import imageio.v3 as iio  # ty:ignore[unresolved-import]
            import matplotlib.pyplot as plt  # ty:ignore[unresolved-import]
            import numpy as np  # ty:ignore[unresolved-import]
            from geom_median.numpy import compute_geometric_median  # ty:ignore[unresolved-import]
            from numpy.typing import NDArray  # ty:ignore[unresolved-import]
            from pacmap import PaCMAP  # ty:ignore[unresolved-import]
        except ImportError:
            print()
            print(
                "[red]Research dependencies not found. Plotting residuals requires "
                "installing Prometheus with the optional research feature, i.e., "
                'using "pip install -U prometheus-llm\\[research]".[/]'
            )
            return

        LAYER_FRAME_MS = 1000
        N_TRANSITION = 20
        TRANSITION_FRAME_MS = 50

        print()
        print("Plotting residual vectors...")

        n_layers = len(self.engine.transformer_layers)
        layer_data_2d = []
        prev_embedding = None

        for li in track(
            range(1, n_layers + 1), description="* Computing PaCMAP projections..."
        ):
            benign_np = self.benign_states[:, li, :].detach().cpu().numpy()
            target_np = self.target_states[:, li, :].detach().cpu().numpy()

            stacked = np.vstack((benign_np, target_np))
            projection = PaCMAP(n_components=2, n_neighbors=30)
            pts_2d = projection.fit_transform(stacked, init=prev_embedding)
            prev_embedding = pts_2d

            n_benign = benign_np.shape[0]
            benign_2d = pts_2d[:n_benign]
            target_2d = pts_2d[n_benign:]

            # Rotate so that benign → target axis is horizontal.
            anchor_b = compute_geometric_median(benign_2d).median
            anchor_t = compute_geometric_median(target_2d).median
            delta = anchor_t - anchor_b
            angle = -np.arctan2(delta[1], delta[0])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            pts_2d = pts_2d @ rot.T

            layer_data_2d.append((pts_2d[:n_benign], pts_2d[n_benign:]))

        plt.style.use(self.config.display.residual_plot_style)
        model_id = self.config.model.model_id

        def _render(
            path: Path,
            layer_idx: int,
            benign_2d: NDArray,
            target_2d: NDArray,
        ):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                benign_2d[:, 0],
                benign_2d[:, 1],
                s=10,
                alpha=0.5,
                c=self.config.benign_prompts.residual_plot_color,
                label=self.config.benign_prompts.residual_plot_label,
            )
            ax.scatter(
                target_2d[:, 0],
                target_2d[:, 1],
                s=10,
                alpha=0.5,
                c=self.config.target_prompts.residual_plot_color,
                label=self.config.target_prompts.residual_plot_label,
            )
            ax.set_title(self.config.display.residual_plot_title, pad=11)
            ax.legend(loc="upper right")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.text(0.018, 0.02, model_id, ha="left", va="bottom", fontsize=12)
            fig.text(
                0.982,
                0.02,
                f"Layer {layer_idx:03}",
                ha="right",
                va="bottom",
                fontsize=12,
            )
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.08)
            fig.savefig(path, dpi=100)
            plt.close(fig)

        out_dir = Path(self.config.display.residual_plot_path) / model_id.replace(
            "/", "_"
        ).replace("\\", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        images = []
        durations = []

        for li, (b2d, t2d) in enumerate(
            track(layer_data_2d, description="* Generating plots..."),
            1,
        ):
            frame_path = out_dir / f"layer_{li:03}.png"
            _render(frame_path, li, b2d, t2d)
            images.append(iio.imread(frame_path))
            durations.append(LAYER_FRAME_MS)

            if li < len(layer_data_2d):
                for fi in range(1, N_TRANSITION):
                    t = fi / N_TRANSITION
                    interp_b = b2d + t * (layer_data_2d[li][0] - b2d)
                    interp_t = t2d + t * (layer_data_2d[li][1] - t2d)
                    tmp = out_dir / f"layer_{li:03}_frame_{fi:03}.png"
                    _render(tmp, li, interp_b, interp_t)
                    images.append(iio.imread(tmp))
                    durations.append(TRANSITION_FRAME_MS)
                    tmp.unlink()

        print("* Generating animation...")
        iio.imwrite(out_dir / "animation.gif", images, duration=durations, loop=0)
        print(f"* Plots saved to [bold]{out_dir.resolve()}[/].")
