import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def save_fig(fig: Figure, output_path: str | Path | None) -> None:
    if output_path is not None:
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        for suf in [".png", ".pdf"]:
            new_path = output_path.with_suffix(suf)

            fig.savefig(new_path, bbox_inches="tight", dpi=300)


# ==========================================
# 1. Mathematical Transformation Functions
# ==========================================

def linear_scaling(x, max_val, clip=True):
    """Linear scaling mapping [0, max_val] to [0, 1]."""
    val = x / max_val
    if clip:
        val = np.clip(val, 0.0, 1.0)
    return val


def global_log_norm(x, ref_scale, clip=True):
    """Base-2 logarithmic compression for absolute residuals."""
    val = np.log2(1 + x / ref_scale)
    if clip:
        val = np.clip(val, 0.0, 1.0)
    return val


def log_norm_derivative(x, ref_scale, clip=True):
    """Analytical derivative of the global log normalization."""
    deriv = 1.0 / (np.log(2) * ref_scale * (1 + x / ref_scale))
    if clip:
        val_unclipped = np.log2(1 + x / ref_scale)
        deriv[val_unclipped > 1.0] = 0.0
    return deriv


# ==========================================
# 2. Clipping Metrics Calculator
# ==========================================

def compute_clipping_metrics(physical_centers, counts, ref_scale):
    """Computes the physical information loss caused by clipping to [-1, 1]."""
    total_pixels = np.sum(counts)

    # A normalized value of 1.0 perfectly corresponds to a physical value of ref_scale
    clip_threshold = ref_scale

    # 1. Saturation Rate
    clipped_mask = physical_centers > clip_threshold
    saturation_rate = np.sum(counts[clipped_mask]) / total_pixels

    # 2. Reconstructed Physical Values (Clipping enforces max physical value of ref_scale)
    reconstructed_centers = np.where(clipped_mask, clip_threshold, physical_centers)

    # 3. Absolute Error Array
    absolute_errors = np.abs(physical_centers - reconstructed_centers)

    # 4. MACE (Mean Absolute Clipping Error / ICE)
    mace = np.sum(counts * absolute_errors) / total_pixels

    # 5. Maximum Truncation Penalty
    valid_bins = physical_centers[counts > 0]
    max_error = np.max(valid_bins) - clip_threshold if len(valid_bins) > 0 and np.max(
        valid_bins) > clip_threshold else 0.0

    return {
        "saturation_rate": saturation_rate,
        "mace": mace,
        "max_error": max_error,
        "absolute_errors": absolute_errors,
        "clip_threshold": clip_threshold
    }


# ==========================================
# 3. Figure Generation Logic
# ==========================================

def generate_normalization_figures(data, output_path=None):
    """Generates a 2x2 analysis figure using absolute residual histogram data."""

    idx = 0
    max_val = data["centered_max"][idx]
    S_ref = data["centered_p98"][idx]
    print(S_ref)
    # S_ref = 76.8

    bin_edges = np.array(data["centered_histogram_bin_edges"][idx])
    counts = np.array(data["centered_histogram_counts"][idx])
    total_pixels = np.sum(counts)

    # Physical bin centers (Absolute residuals: domain is 0 to +infinity)
    physical_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute the theoretical clipping metrics
    metrics = compute_clipping_metrics(physical_centers, counts, S_ref)

    # Map to normalized space
    lin_centers = linear_scaling(physical_centers, max_val, clip=True)
    log_centers = global_log_norm(physical_centers, S_ref, clip=True)

    # Re-bin into visual space [0, 1.1] to clearly show the pile-up at 1.0
    vis_bins = np.linspace(0, 1.1, 100)
    vis_widths = np.diff(vis_bins)

    lin_counts, _ = np.histogram(lin_centers, bins=vis_bins, weights=counts)
    lin_density = lin_counts / (total_pixels * vis_widths)

    log_counts, _ = np.histogram(log_centers, bins=vis_bins, weights=counts)
    log_density = log_counts / (total_pixels * vis_widths)

    # -- Plotting Setup --
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--'
    })

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    color_lin = '#1f77b4'
    color_log = '#d62728'

    # --- (a) Transfer Function ---
    plot_limit = min(max_val, S_ref * 4)  # Focus on the active region
    x_domain = np.linspace(0, plot_limit, 1000)

    y_lin = linear_scaling(x_domain, max_val, clip=False)
    y_log_clipped = global_log_norm(x_domain, S_ref, clip=True)
    y_log_unclipped = global_log_norm(x_domain, S_ref, clip=False)

    axs[0, 0].plot(x_domain, y_lin, label=f'Linear (Max={max_val:.1f}m)', color=color_lin, lw=2)
    axs[0, 0].plot(x_domain, y_log_unclipped, label='Global Log (Unclipped)', color=color_log, lw=2, ls=':')
    axs[0, 0].plot(x_domain, y_log_clipped, label='Global Log (Clipped)', color=color_log, lw=2)

    axs[0, 0].axhline(1, color='black', lw=1, ls='--')
    axs[0, 0].set_title("(a) Transfer Function (Absolute Domain)")
    axs[0, 0].set_xlabel("Absolute Physical Residual (m)")
    axs[0, 0].set_ylabel("Normalized Value")
    axs[0, 0].set_ylim(0, 1.2)
    axs[0, 0].legend()

    # --- (b) Encoded Data Distribution ---
    axs[0, 1].stairs(lin_density, vis_bins, fill=True, alpha=0.6, color=color_lin, label='Linear Scaling')
    axs[0, 1].stairs(log_density, vis_bins, fill=True, alpha=0.6, color=color_log, label='Global Log')

    axs[0, 1].set_title("(b) Probability Density of Encoded Data")
    axs[0, 1].set_xlabel("Absolute Normalized Value")
    axs[0, 1].set_ylabel("Density (Log Scale)")
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlim(0, 1.15)

    axs[0, 1].axvline(1, color='black', lw=1, ls='--', alpha=0.8, label="Clipping Boundary")
    axs[0, 1].legend()

    # --- (c) Expansion Ratio ---
    deriv_lin = np.full_like(x_domain, 1.0 / max_val)
    deriv_log = log_norm_derivative(x_domain, S_ref, clip=True)

    safe_deriv_lin = np.where(deriv_lin == 0, 1e-12, deriv_lin)
    expansion_ratio = deriv_log / safe_deriv_lin
    expansion_ratio[(deriv_log == 0)] = 0.0

    axs[1, 0].plot(x_domain, expansion_ratio, color='purple', lw=2, label='Log / Linear Gradient Ratio')
    axs[1, 0].axhline(1, color='black', lw=1.5, ls='--', label='Baseline (1:1 Ratio)')

    valid_expansion = (expansion_ratio > 1) & (deriv_log > 0)
    axs[1, 0].fill_between(x_domain, 1, expansion_ratio, where=valid_expansion,
                           color='purple', alpha=0.1, label='Expansion Zone (>1.0)')

    axs[1, 0].axvline(metrics["clip_threshold"], color=color_log, lw=1, ls='--', label='Log Clip Threshold')

    axs[1, 0].set_title("(c) Low-Relief Representation Gain")
    axs[1, 0].set_xlabel("Absolute Physical Residual (m)")
    axs[1, 0].set_ylabel("Gradient Multiplier vs Linear")
    axs[1, 0].set_yscale('symlog', linthresh=0.1)
    axs[1, 0].set_ylim(-0.1, max(10, np.max(expansion_ratio) * 1.2))
    axs[1, 0].legend()

    # --- (d) Clipping Penalty Analysis ---
    axs[1, 1].plot(physical_centers, metrics["absolute_errors"], color='darkorange', lw=2, label='Information Loss (m)')

    # Shaded region indicating the active, unclipped zone
    axs[1, 1].axvspan(0, metrics["clip_threshold"], color='green', alpha=0.1, label='Safely Encoded Region')

    axs[1, 1].set_title("(d) Truncation Penalty Profile")
    axs[1, 1].set_xlabel("Absolute Physical Residual (m)")
    axs[1, 1].set_ylabel("Reconstruction Error (m)")
    axs[1, 1].set_xlim(0, plot_limit)
    axs[1, 1].set_ylim(0, max(1, np.max(metrics["absolute_errors"][physical_centers < plot_limit]) * 1.2))
    axs[1, 1].legend()

    # Overlay computed metrics text box
    textstr = '\n'.join((
        r'$\bf{Clipping\ Metrics}$',
        f'Threshold ($S_{{ref}}$): {metrics["clip_threshold"]:.2f} m',
        f'Saturation Rate: {metrics["saturation_rate"] * 100:.3f}%',
        f'ICE (MACE): {metrics["mace"]:.4f} m',
        f'Max Penalty: {metrics["max_error"]:.2f} m'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    axs[1, 1].text(0.55, 0.25, textstr, transform=axs[1, 1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)

    if output_path:
        save_fig(fig, output_path)
        print(f"Figure successfully written to {output_path}")

        # Print metrics to terminal as well
        print("\n--- Clipping Assessment ---")
        print(f"Saturation Rate (Data lost to tails):  {metrics['saturation_rate'] * 100:.4f}%")
        print(f"Irreducible Clipping Error (MACE):     {metrics['mace']:.5f} meters")
        print(f"Maximum Truncation Penalty:            {metrics['max_error']:.2f} meters")
    else:
        plt.show()


def compute_sweep_metrics(physical_centers, counts, s_ref_candidates, max_val):
    """Sweeps through candidate S_ref values and computes the Cost/Benefit."""
    total_pixels = np.sum(counts)

    mace_list = []
    expansion_list = []

    linear_gradient = 1.0 / max_val if max_val > 0 else 1.0

    for s_ref in s_ref_candidates:
        # Cost: Mean Absolute Clipping Error
        clipped_mask = physical_centers > s_ref
        reconstructed = np.where(clipped_mask, s_ref, physical_centers)
        errors = np.abs(physical_centers - reconstructed)
        mace = np.sum(counts * errors) / total_pixels

        # Benefit: Expansion Factor (Evaluated near 0 for stability)
        x_eval = 0.1
        log_gradient = 1.0 / (np.log(2) * s_ref * (1 + x_eval / s_ref))
        expansion_factor = log_gradient / linear_gradient

        mace_list.append(mace)
        expansion_list.append(expansion_factor)

    return np.array(mace_list), np.array(expansion_list)


def generate_optimization_figures(data, output_path=None):
    idx = 0
    max_val = data["centered_max"][idx]

    bin_edges = np.array(data["centered_histogram_bin_edges"][idx])
    counts = np.array(data["centered_histogram_counts"][idx])
    physical_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Use logspace for even distribution on a log-log plot
    s_ref_candidates = np.logspace(np.log10(2.0), np.log10(max_val * 0.9), 1000)

    mace, expansion = compute_sweep_metrics(
        physical_centers, counts, s_ref_candidates, max_val
    )

    # --- Constrained Optimization Logic ---
    # Define the strict physical tolerance limit (e.g., 10 cm average error)
    max_acceptable_mace = 0.10

    # Find the smallest S_ref that satisfies the physical constraint
    valid_indices = np.where(mace <= max_acceptable_mace)[0]
    if len(valid_indices) == 0:
        optimal_idx = np.argmin(mace)
    else:
        optimal_idx = valid_indices[0]

    optimal_s_ref = s_ref_candidates[optimal_idx]
    optimal_mace = mace[optimal_idx]
    optimal_exp = expansion[optimal_idx]

    # -- Plotting Setup --
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
        'legend.fontsize': 11, 'axes.grid': True, 'grid.alpha': 0.4, 'grid.linestyle': '--'
    })

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # =========================================================
    # (a) Hyperparameter Sweep (Dual Log-Log Axis)
    # =========================================================
    ax1 = axs[0]
    ax2 = ax1.twinx()

    l1, = ax1.plot(s_ref_candidates, expansion, color='purple', lw=2.5, label='Benefit: Expansion Factor')
    l2, = ax2.plot(s_ref_candidates, mace, color='darkorange', lw=2.5, label='Cost: MACE (m)')

    ax1.axvline(optimal_s_ref, color='black', lw=1.5, ls='--',
                label=rf'Optimal $S_{{ref}}$ = {optimal_s_ref:.1f}m\n(MACE $\leq$ {max_acceptable_mace}m)')

    # Set log scales for all three axes
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_title("(a) Hyperparameter Trade-off Sweep")
    ax1.set_xlabel("Candidate $S_{ref}$ Threshold (m) [Log Scale]")
    ax1.set_ylabel("Expansion Ratio vs Linear [Log Scale]", color='purple', fontweight='bold')
    ax2.set_ylabel("Mean Absolute Clipping Error (m) [Log Scale]", color='darkorange', fontweight='bold')

    ax1.tick_params(axis='y', colors='purple')
    ax2.tick_params(axis='y', colors='darkorange')

    lines = [l1, l2, ax1.lines[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)

    # =========================================================
    # (b) The Pareto Frontier (Log-Log)
    # =========================================================
    axs[1].plot(mace, expansion, color='#1f77b4', lw=2.5)

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    # Highlight the constrained optimum
    axs[1].scatter(optimal_mace, optimal_exp, color='red', s=100, zorder=5,
                   label=f'Constrained Optimum ($S_{{ref}}$={optimal_s_ref:.1f}m)')

    # Add a shaded warning zone for physically unacceptable errors
    # On log scale, we must give a definitive max bound for the span
    axs[1].axvspan(max_acceptable_mace, np.max(mace) * 2, color='red', alpha=0.1, label='Unacceptable Error Zone')

    axs[1].set_title("(b) Constrained Optimization Pareto Frontier")
    axs[1].set_xlabel("Cost: Mean Absolute Clipping Error (m) [Log Scale]")
    axs[1].set_ylabel("Benefit: Base Expansion Factor [Log Scale]")

    axs[1].set_xlim(np.min(mace) * 0.8, np.max(mace) * 1.2)
    axs[1].set_ylim(np.min(expansion) * 0.8, np.max(expansion) * 1.2)

    # Annotate specific acceptable tolerance levels
    # textcoords='offset points' ensures text doesn't overlap the line regardless of log scaling
    for tol in [0.1, 0.5, 1.0]:
        idx_tol = np.argmin(np.abs(mace - tol))

        # We don't need a strict absolute difference check on a log scale,
        # just check if it's reasonably close ratio-wise
        if np.abs(mace[idx_tol] - tol) / tol < 0.2:
            s_val = s_ref_candidates[idx_tol]
            axs[1].scatter(mace[idx_tol], expansion[idx_tol], color='black', s=40, zorder=4)
            axs[1].annotate(f' Tol={tol}m\n $S_{{ref}}$={s_val:.1f}m',
                            (mace[idx_tol], expansion[idx_tol]),
                            xytext=(-8, -12), textcoords='offset points',
                            fontsize=10, ha="right")

    axs[1].legend()

    if output_path:
        save_fig(fig, output_path)
        print(f"Figure successfully written to {output_path}")
    else:
        plt.show()


# ==========================================
# 3. Integration with CLI
# ==========================================

def main():
    input_json = "../dataset_stats/dtm/dataset_stats.json"
    parser = argparse.ArgumentParser(description="Generate LaTeX table and figures from dataset statistics JSON.")
    parser.add_argument("--out-folder", type=str, default="../outputs/figures/",
                        help="Optional output .pdf or .png file for the figure")

    args = parser.parse_args()

    with open(input_json, "r") as f:
        data = json.load(f)

    path = Path(args.out_folder)

    path.mkdir(parents=True, exist_ok=True)

    # 2. Output the Figure
    generate_normalization_figures(data, path / "normalization.png")
    generate_optimization_figures(data, path / "optimization.png")


if __name__ == "__main__":
    main()
