import pandas as pd

from cco.problems.problem import Problem
from cco.utils import *
from cco.utils.eval import get_results_from_histories
from cco.utils.eval import get_stats
from cco.utils.file_utils import get_file_path, get_save_path
from cco.utils.run import run_algorithm


base_dir = "output/data/"


def run_experiment(
    problem: Problem, methods: list, repeat: int, ec_samples_num: int = 100000, parallel: bool = True, verbose: bool = True
):
    df_stats = pd.DataFrame()
    for method in methods:
        save_path = get_save_path(problem, content="eval", method=method, repeat=repeat)
        x_histories, ec_histories, run_times = run_algorithm(
            problem=problem,
            method=method,
            repeat=repeat,
            ec_samples_num=ec_samples_num,
            save_results=False,
            verbose=verbose,
            parallel=parallel,
        )
        results = get_results_from_histories(
            problem=problem,
            method=method,
            x_histories=x_histories,
            ec_histories=ec_histories,
            run_times=run_times,
            save_path=save_path,
        )

        print("=" * 100)
        print(results.drop(columns=["lr", "theta", "mu", "eps"]))

        save_path = get_save_path(
            problem, content="method_stats", repeat=repeat, method=method
        )
        stats = get_stats(results)
        df_stats[method] = pd.Series(stats)
        df_stats[method].to_csv(get_file_path(save_path, base_dir=base_dir))
        print("=" * 100)
        print(df_stats[method])
        print("=" * 100)

    save_path = get_save_path(problem, content="stats", repeat=repeat)
    df_stats.T.to_csv(get_file_path(save_path, base_dir=base_dir))
