from pathlib import Path
from cco.problems.problem import Problem

def get_save_path(
    problem: Problem, content: str, method: str = None, repeat: int = None
) -> str:
    if method is not None and method.lower() == "core":
        if content == "data":
            save_path = f"{problem.name}/{method.lower()}/runs/initial_x={problem.initial_x}_lr={problem.lr}_theta={problem.theta}_eps={problem.epsilon}_mu={problem.mu}_max_iter={problem.max_iter_x}_max_iter_s={problem.max_iter_s}.json"
        elif content == "plot":
            save_path = f"{problem.name}/{method.lower()}/initial_x={problem.initial_x}_lr={problem.lr}_theta={problem.theta}_eps={problem.epsilon}_mu={problem.mu}_max_iter={problem.max_iter_x}_max_iter_s={problem.max_iter_s}.png"
        elif content == "plot_std":
            save_path = f"{problem.name}/{method.lower()}/std_repeat={repeat}_initial_x={problem.initial_x}_lr={problem.lr}_theta={problem.theta}_eps={problem.epsilon}_mu={problem.mu}_max_iter={problem.max_iter_x}_max_iter_s={problem.max_iter_s}.png"
        elif content == "eval":
            save_path = f"{problem.name}/{method.lower()}/repeat={repeat}_samples_num={problem.samples_num}_initial_x={problem.initial_x}_lr={problem.lr}_theta={problem.theta}_eps={problem.epsilon}_mu={problem.mu}_max_iter={problem.max_iter_x}_max_iter_s={problem.max_iter_s}.csv"
        elif content == "method_stats":
            save_path = f"{problem.name}/{method.lower()}/method_stats_repeat={repeat}_samples_num={problem.samples_num}_initial_x={problem.initial_x}_lr={problem.lr}_theta={problem.theta}_mu={problem.mu}_max_iter={problem.max_iter_x}_max_iter_s={problem.max_iter_s}.csv"
    elif method is not None and method.lower() in ["cpp-mip", "cpp-kkt"]:
        if content == "eval":
            save_path = f"{problem.name}/{method.lower()}/repeat={repeat}_samples_num={problem.samples_num}.csv"
        elif content == "method_stats":
            save_path = f"{problem.name}/{method.lower()}/method_stats_repeat={repeat}_samples_num={problem.samples_num}.csv"
    elif content == "stats":
        save_path = f"{problem.name}/stats_repeat={repeat}_samples_num={problem.samples_num}.csv"
    elif content == "grid_search_stats":
        save_path = f"{problem.name}/core/grid_search_stats_repeat={repeat}_samples_num={problem.samples_num}.csv"

    return save_path

def get_unique_path(file_path):
    file_path = Path(file_path)
    base = file_path.stem
    ext = file_path.suffix
    parent = file_path.parent
    candidate = file_path
    i = 2
    while candidate.exists():
        candidate = parent / f"{base}_{i}{ext}"
        i += 1
    return candidate


def get_file_path(save_path, base_dir, unique=True):
    # Always save under output/plots, ignore any leading slashes or absolute paths
    save_name = str(save_path).lstrip("/\\")  # Remove leading / or \
    file_path = Path(base_dir) / save_name
    if file_path.suffix == "":
        raise ValueError(
            "save_path must have a file extension (e.g., .png, .json). filename not specified."
        )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return get_unique_path(file_path) if unique else file_path
