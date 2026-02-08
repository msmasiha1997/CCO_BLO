from cco.problems.problem import Problem
from cco.utils.file_utils import get_file_path


import numpy as np
import pandas as pd

base_dir = "output/data/"

def get_results_from_histories(
    problem: Problem,
    method: str,
    x_histories: np.ndarray,
    ec_histories: np.ndarray,
    run_times: float,
    save_path: str | None = None,
):

    xs = [x_history[-1] for x_history in x_histories]
    ecs = [ec_history[-1] for ec_history in ec_histories]
    objective_values = [float(problem.f_function(x_history[-1])) for x_history in x_histories]
    results = pd.DataFrame(
        {
            "method": method,
            "lr": float(problem.lr) if method == "Core" else None,
            "theta": float(problem.theta) if method == "Core" else None,
            "mu": float(problem.mu) if method == "Core" else None,
            "eps": float(problem.epsilon) if method == "Core" else None,
            "objective_values": objective_values,
            "xs": xs,
            "empirical_coverages": ecs,
            "run_times": run_times,
        }
    )

    if save_path:
        file_path = get_file_path(save_path, base_dir)
        results.to_csv(file_path, index=False)

    return results


def get_stats(results: pd.DataFrame):
    x_mean = np.mean(results["xs"])
    x_std = np.std(np.array(results["xs"]), axis=0)
    obj_mean = np.mean(results["objective_values"])
    obj_std = np.std(results["objective_values"])
    ec_mean = np.mean(results["empirical_coverages"])
    ec_std = np.std(results["empirical_coverages"])
    run_times_mean = np.mean(results["run_times"])

    stats = {
        "lr": results["lr"].iloc[0],
        "theta": results["theta"].iloc[0],
        "mu": results["mu"].iloc[0],
        "eps": results["eps"].iloc[0],
        "obj_mean": obj_mean,
        "obj_std": obj_std,
        "x_mean": x_mean,
        "x_std": x_std,
        "ec_mean": ec_mean,
        "ec_std": ec_std,
        "run_times_mean": run_times_mean,
    }

    return pd.Series(stats)