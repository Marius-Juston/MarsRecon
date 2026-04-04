import json
import argparse

def compute_metrics(data):
    channels = data.get("channels", ["Near-infrared", "Red", "Blue-green"])
    n_pixels = data["n_valid_pixels_per_channel"]
    mean = data["mean"]
    std = data["std"]
    min_ = data["min"]
    max_ = data["max"]
    p02 = data["p02"]
    p98 = data["p98"]

    results = []

    for i in range(len(n_pixels)):
        mu = mean[i]
        sigma = std[i]

        cv = sigma / mu if mu != 0 else float("nan")
        drcr = (p98[i] - p02[i]) / mu if mu != 0 else float("nan")

        channel_name: str = channels[i] if i < len(channels) else f"Channel {i}"
        channel_name = channel_name.capitalize()

        results.append({
            "channel": channel_name,
            "n_pixels": n_pixels[i],
            "mean": mu,
            "std": sigma,
            "min": min_[i],
            "max": max_[i],
            "p02": p02[i],
            "p98": p98[i],
            "cv": cv,
            "drcr": drcr
        })

    return results


def format_latex_table(results):
    header = r"""\begin{table}[t]
\centering
\caption{Summary statistics of the dataset across spectral channels. P2 and P98 represent the 2\% and 98\% percentiles. CV denotes coefficient of variation ($\sigma/\mu$). DRCR denotes dynamic range compression ratio ($(P_{98}-P_{2})/\mu$).}
\label{tab:dataset_stats_extended}

% --- Block 1: Basic Statistics ---
\begin{tabular}{
l 
S[table-format=1.4e2] 
S[table-format=1.4] 
S[table-format=1.4]
}
\toprule
\textbf{Channel} & {\textbf{$N_{\text{pixels}}$}} & {\textbf{Mean}} & {\textbf{Std}} \\
\midrule
"""

    rows1 = ""
    for r in results:
        rows1 += (
            f"{r['channel']} & "
            f"{r['n_pixels']:.4e} & "
            f"{r['mean']:.4f} & "
            f"{r['std']:.4f} \\\\\n"
        )

    mid_section = r"""\bottomrule
\end{tabular}

\vspace{1.5em} 

% --- Block 2: Extrema & Derived Metrics ---
\begin{tabular}{
l 
S[table-format=1.4] 
S[table-format=1.4] 
S[table-format=1.4] 
S[table-format=1.4] 
S[table-format=1.3]
S[table-format=1.3]
}
\toprule
\textbf{Channel} & {\textbf{Min}} & {\textbf{Max}} & {\textbf{P2}} & {\textbf{P98}} & {\textbf{CV}} & {\textbf{DRCR}} \\
\midrule
"""

    rows2 = ""
    for r in results:
        rows2 += (
            f"{r['channel']} & "
            f"{r['min']:.4f} & "
            f"{r['max']:.4f} & "
            f"{r['p02']:.4f} & "
            f"{r['p98']:.4f} & "
            f"{r['cv']:.3f} & "
            f"{r['drcr']:.3f} \\\\\n"
        )

    footer = r"""\bottomrule
\end{tabular}
\end{table}
"""

    return header + rows1 + mid_section + rows2 + footer


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from dataset statistics JSON.")
    parser.add_argument("input_json", type=str, help="Path to input JSON file")
    parser.add_argument("--output", type=str, default=None, help="Optional output .tex file")

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    results = compute_metrics(data)
    latex_table = format_latex_table(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(latex_table)
        print(f"Table successfully written to {args.output}")
    else:
        print(latex_table)


if __name__ == "__main__":
    main()