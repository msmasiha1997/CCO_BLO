import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cco.core.oracle import Oracle
from cco.problems.problem import Problem
from cco.utils.file_utils import get_file_path, get_unique_path

base_dir = "output/plots/"

# Optional seaborn import for high-contrast palettes
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def _distinct_colors(n: int):
    """Return n visually distinct colors.

    Prefers seaborn's "husl" palette for arbitrarily many distinct hues.
    Falls back to concatenated tab20/tab20b/tab20c (up to 60), and finally HSV.
    """
    if n <= 0:
        return []
    if _HAS_SEABORN:
        return sns.color_palette("husl", n)

    # Fallback: stitch discrete qualitative palettes for more variety
    palettes = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(name)
        # Prefer discrete list if available
        if hasattr(cmap, "colors"):
            palettes.extend(list(cmap.colors))
        else:
            palettes.extend(list(cmap(np.linspace(0, 1, 20))))

    if n <= len(palettes):
        # Spread selections across the stitched palette to maximize contrast
        idxs = np.linspace(0, len(palettes) - 1, n).astype(int)
        return [palettes[i] for i in idxs]

    # Last resort: evenly spaced hues in HSV space
    return [plt.cm.hsv(i / n) for i in range(n)]


def plot_objective_function(
    problem: Problem,
    x_history: np.ndarray,
    ec_history: np.ndarray,
    save_path=None,
    figsize=(12, 6),
    suboptimality: bool = False,
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"Problem: {problem.name}", fontsize=16)
    if suboptimality:
        suboptimality = Oracle.suboptimality_f(problem, x_history)
        iterations = np.arange(len(suboptimality))
        ax1.plot(iterations, suboptimality, label="Suboptimality")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Sub-optimality")
        ax1.set_title("Sub-optimality of the optimal value")
        ax1.grid(True)
    else:
        optimality = problem.f_function(x_history)
        iterations = np.arange(len(optimality))
        ax1.plot(iterations, optimality, label="Optimality")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Objective function value")
        ax1.set_title("Optimality of the objective function value")
        ax1.grid(True)

    ax2.plot(iterations, ec_history, label="Empirical coverage")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical Coverage")
    ax2.set_title("Empirical coverage of x_history")
    target_coverage = 1 - problem.delta
    ax2.axhline(
        y=target_coverage,
        color="red",
        linestyle="--",
        label=f"target ({target_coverage:.3f})",
    )
    ax2.grid(True)

    fig.tight_layout()
    if save_path:
        file_path = get_file_path(save_path, base_dir)
        try:
            plt.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Could not save the plot to {file_path}: {e}")


def plot_std(
    problem: Problem,
    x_histories: list[np.ndarray],
    ec_histories: list[np.ndarray],
    save_path: str = None,
    figsize=(12, 6),
    obj_lim:tuple[float, float] = None,
    ec_lim: tuple[float, float] = None,
):
    # Input should be padded
    obj_histories = [problem.f_function(x_history) for x_history in x_histories]
    obj_histories = np.array(obj_histories)
    ec_histories = np.array(ec_histories)


    mean_obj = np.mean(obj_histories, axis=0).reshape(-1)
    std_obj = np.std(obj_histories, axis=0).reshape(-1)
    mean_ec = np.mean(ec_histories, axis=0)
    std_ec = np.std(ec_histories, axis=0)
    x = np.arange(len(mean_obj))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(x, mean_obj, label="Mean Obj", color="blue")
    # print("mean_obj:", mean_obj)
    # print("std_obj:", std_obj)
    # print("x:", x)
    ax1.fill_between(x, mean_obj - std_obj, mean_obj + std_obj, color="blue", alpha=0.2, label="Std Dev Obj")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Objective function value")
    ax1.set_title("Mean and Std of Objective Values")
    if obj_lim:
        ax1.set_ylim(obj_lim)
    ax1.legend()
    ax1.grid(True)

    # Plot ec_histories mean and std
    ax2.plot(x, mean_ec, label="Mean EC", color="green")
    ax2.fill_between(x, mean_ec - std_ec, mean_ec + std_ec, color="green", alpha=0.2, label="Std Dev EC")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical Coverage")
    ax2.set_title("Mean and Std of Empirical Coverage")
    ax2.axhline(y=1 - problem.delta, color="red", linestyle="--", label="Target")
    if ec_lim:
        ax2.set_ylim(ec_lim)
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    if save_path:
        file_path = get_file_path(save_path, base_dir)
        try:
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Could not save the plot to {file_path}: {e}")
    plt.show()

def plot_compare_stds(
        problem: Problem,
        x_histories_list: list[list[np.ndarray]],
        ec_histories_list: list[list[np.ndarray]],
        labels: list[str] | None = None,
        save_path: str = None,
        figsize=(12, 6),
        obj_lim:tuple[float, float] = None,
        ec_lim: tuple[float, float] = None,
):
    obj_histories_list = [[problem.f_function(x_history).reshape(-1) for x_history in x_histories] for x_histories in x_histories_list]
    iter_nums = [np.arange(len(x_histories[0])) for x_histories in x_histories_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"Problem: {problem.name}, samples_num={problem.samples_num}", fontsize=16)

    n = len(obj_histories_list)  # number of series
    colors = _distinct_colors(n)

    for i, (obj_histories, iters) in enumerate(zip(obj_histories_list, iter_nums)):
        mean_obj = np.mean(obj_histories, axis=0)
        std_obj = np.std(obj_histories, axis=0)
        ax1.plot(iters, mean_obj, label=labels[i] if labels else f"Run {i}", color=colors[i])
        ax1.fill_between(iters, mean_obj - std_obj, mean_obj + std_obj, color=colors[i], alpha=0.2)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Objective function value")
    ax1.set_title("Mean and Std of Objective Values")
    if obj_lim:
        ax1.set_ylim(obj_lim)
    ax1.grid(True)
    ax1.legend()

    for i, (ec_histories, iters) in enumerate(zip(ec_histories_list, iter_nums)):
        mean_ec = np.mean(ec_histories, axis=0)
        std_ec = np.std(ec_histories, axis=0)
        ax2.plot(iters, mean_ec, label=labels[i] if labels else f"Run {i}", color=colors[i])
        ax2.fill_between(iters, mean_ec - std_ec, mean_ec + std_ec, color=colors[i], alpha=0.2)

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical Coverage")
    ax2.set_title("Mean and Std of Empirical Coverage")
    ax2.axhline(y=1 - problem.delta, color="red", linestyle="--", label="Target")
    if ec_lim:
        ax2.set_ylim(ec_lim)
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    if save_path:
        file_path = get_file_path(save_path, base_dir)
        try:
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Could not save the plot to {file_path}: {e}")

def plot_compare_runs(
    problem: Problem,
    x_histories: list[np.ndarray],
    ec_histories: list[np.ndarray],
    labels: list[str] | None = None,
    save_path=None,
    figsize=(12, 6),
    obj_lim: tuple[float, float] = None,
    ec_lim: tuple[float, float] = None
):
    optimalities = [problem.f_function(x_history) for x_history in x_histories]
    iter_nums = [np.arange(len(x_history)) for x_history in x_histories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Problem: {problem.name}", fontsize=16)

    for i, (opt, iters) in enumerate(zip(optimalities, iter_nums)):
        ax1.plot(iters, opt, label=labels[i] if labels else f"Run {i}")

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Objective function value")
    ax1.set_title("Objective function value comparison")
    if obj_lim:
        ax1.set_ylim(obj_lim)
    ax1.grid(True)
    ax1.legend()

    for i, (ec_history, iters) in enumerate(zip(ec_histories, iter_nums)):
        ax2.plot(iters, ec_history, label=labels[i] if labels else f"Run {i}")

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical Coverage")
    ax2.set_title("Empirical Coverage comparison")
    ax2.axhline(y=1 - problem.delta, color="red", linestyle="--", label="Target")
    if ec_lim:
        ax2.set_ylim(ec_lim)
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    if save_path:
        file_path = get_file_path(save_path, base_dir)
        try:
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Could not save the plot to {file_path}: {e}")


def plot_compare_two(
    problem: Problem,
    x_history_1: np.ndarray,
    ec_history_1: np.ndarray,
    x_history_2: np.ndarray,
    ec_history_2: np.ndarray,
    labels=("run1", "run2"),
    save_path: str | None = None,
    figsize=(12, 6),
):
    """Compare two optimization runs.

    Left subplot: suboptimality for both runs.
    Right subplot: empirical-coverage trajectories for both runs.
    """

    opt1 = problem.f_function(x_history_1)
    opt2 = problem.f_function(x_history_2)

    iters1 = np.arange(len(opt1))
    iters2 = np.arange(len(opt2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"Problem: {problem.name}", fontsize=16)

    ax1.plot(iters1, opt1, label=labels[0], linewidth=2)
    ax1.plot(iters2, opt2, label=labels[1], linewidth=2)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Objective function value")
    ax1.set_title("Objective function value comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    max_len = max(len(ec_history_1), len(ec_history_2))
    iters_ec1 = np.arange(len(ec_history_1))
    iters_ec2 = np.arange(len(ec_history_2))

    ax2.plot(iters_ec1, ec_history_1, label=labels[0], linewidth=2)
    ax2.plot(iters_ec2, ec_history_2, label=labels[1], linewidth=2)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical coverage")
    ax2.set_title("Empirical coverage comparison")
    target_coverage = 1 - problem.delta
    ax2.axhline(
        y=target_coverage,
        color="red",
        linestyle="--",
        label=f"target ({target_coverage:.3f})",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    if save_path:
        file_path = get_file_path(save_path, base_dir)
        try:
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Could not save comparison plot to {file_path}: {e}")
