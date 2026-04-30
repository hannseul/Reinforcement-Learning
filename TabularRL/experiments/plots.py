import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_learning_curves(curves, title, xlabel, ylabel, save_path=None):
    """
    curves: dict name -> list/array of y-values oder list von (x, y)-Tupeln.
    """
    plt.figure(figsize=(8, 5))
    for name, values in curves.items():
        values = list(values)
        if len(values) == 0:
            continue
        if isinstance(values[0], tuple):
            x = [v[0] for v in values]
            y = [v[1] for v in values]
        else:
            x = list(range(len(values)))
            y = values
        plt.plot(x, y, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
    return plt.gcf()


def plot_bias_summary(bias_results, save_path=None):
    """
    bias_results kommt aus metrics.bias_metrics.compare_biases.
    """
    names = list(bias_results.keys())
    total_bias = [bias_results[n]["summed_total_bias"] for n in names]
    squared_bias = [bias_results[n]["summed_squared_total_bias"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, total_bias, width, label="summed total bias")
    plt.bar(x + width / 2, squared_bias, width, label="summed squared total bias")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.title("Overestimation Bias Vergleich")
    plt.ylabel("Bias-Metrik")
    plt.legend()
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
    return plt.gcf()


def plot_policy_grid(policy, rows, cols, title="Policy", save_path=None):
    arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→", None: "T"}
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.grid(True)
    ax.set_title(title)
    ax.invert_yaxis()

    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            action = policy.get(state)
            ax.text(c + 0.5, r + 0.5, arrows.get(action, str(action)), ha="center", va="center", fontsize=16)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
    return fig
