import os
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial

from cco.problems.problem import Problem
from cco.core.optimizer import Optimizer
from cco.core.oracle import Oracle
from cco.taco.problems.toy_problem import ToyProblem
from cco.taco.chance_optimizer import Optimizer as TacoOptimizer
from cco.utils import *
from cco.utils.plotting import *
from cco.utils.file_utils import *
from cco.utils.eval import *
from cco.cpp.resources.solver import solve


base_dir = "output/data/"


def grid_search_core(
    problem: Problem,
    thetas: list = None,
    epsilons: list = None,
    lrs: list = None,
    mus: list = None,
    ec_samples_num: int = 100000,
    error: float = 0.005,
    repeat: int = 1,
    obj_lim: tuple[float, float] = None,
    ec_lim: tuple[float, float] = None,
    save_plots: bool = True,
    save_results: bool = True,
    parallel: bool = False,
    n_jobs: int = None,
):

    if thetas is None:
        thetas = [problem.theta]
    if epsilons is None:
        epsilons = [problem.epsilon]
    if lrs is None:
        lrs = [problem.lr]
    if mus is None:
        mus = [problem.mu]

    if save_plots:
        x_histories_list = []
        ec_histories_list = []
    x_histories = []
    ec_histories = []
    labels = []

    best_x = None
    best_ec = float("-inf")
    best_i = -1
    best_opt = float("inf")
    i = 0
    df_stats = pd.DataFrame() if save_results else None
    print("Grid Search Core:")
    for theta in thetas:
        for epsilon in epsilons:
            for mu in mus:
                for lr in lrs:
                    if save_results:
                        eval_path = get_save_path(problem, content="eval", repeat=repeat, method="Core")
                        stats_path = get_save_path(problem, content="method_stats", repeat=repeat, method="Core")
                        results_path = get_save_path(problem, content="data", method="Core")
                        plots_path = get_save_path(problem, content="plot", method="Core")
                        plot_stds_path = get_save_path(problem, content="plot_std", method="Core")
                    else:
                        eval_path, stats_path, results_path, plots_path, plot_stds_path = None, None, None, None, None
                    problem.theta = theta
                    problem.epsilon = epsilon
                    problem.lr = lr
                    problem.mu = mu
                    if repeat == 1:
                        x_history, ec_history, run_time = run_core(
                            problem,
                            ec_samples_num=ec_samples_num,
                            verbose=False,
                            save_path=results_path,
                        )
                    else:
                        _x_histories, _ec_histories, _ = run_algorithm(
                            problem=problem,
                            method="Core",
                            repeat=repeat,
                            ec_samples_num=ec_samples_num,
                            verbose=False,
                            parallel=parallel,
                            n_jobs=n_jobs,
                        )

                        if save_results:
                            results = get_results_from_histories(
                                problem=problem,
                                method="Core",
                                x_histories=_x_histories,
                                ec_histories=_ec_histories,
                                run_times=_,
                                save_path=eval_path,
                            )
                            stats = get_stats(results)
                            df_stats[i] = pd.Series(stats)
                            df_stats[i].to_csv(
                                get_file_path(stats_path, base_dir=base_dir)
                            )

                        _x_histories = pad_histories(_x_histories)
                        _ec_histories = pad_histories(_ec_histories)
                        x_history = np.mean(_x_histories, axis=0)
                        ec_history = np.mean(_ec_histories, axis=0)
                        x_histories_list.append(_x_histories)
                        ec_histories_list.append(_ec_histories)

                    if repeat == 1:
                        plot_objective_function(
                            problem=problem,
                            x_history=x_history,
                            ec_history=ec_history,
                            save_path=plots_path,
                            obj_lim=obj_lim,
                            ec_lim=ec_lim,
                        )
                    else:
                        plot_std(
                            problem=problem,
                            x_histories=_x_histories,
                            ec_histories=_ec_histories,
                            save_path=plot_stds_path,
                            obj_lim=obj_lim,
                            ec_lim=ec_lim,
                        )

                    x_histories.append(x_history)
                    ec_histories.append(ec_history)
                    labels.append(f"lr={lr},theta={theta},mu={mu}")
                    x = x_history[-1]
                    ec = ec_history[-1]
                    opt = problem.f_function(x)

                    if opt <= best_opt and (
                        ec >= 1 - problem.delta - error or ec >= best_ec
                    ):
                        best_opt = opt
                        best_ec = ec
                        best_x = x
                        best_i = i
                    print(f"{i}: epsilon={epsilon}, theta={theta}, lr={lr}, mu={mu}")
                    print(f"{i}: ec={ec:.8f}, obj={float(opt):8f}")
                    print(f"{i}: x={x}")
                    i += 1

    print("=" * 40)
    print(f"Best optimality: {float(best_opt):.10f}")
    print(f"Best empirical coverage: {float(best_ec):.10f}")
    print(f"Best x: {best_x}")
    print(f"Best i: {best_i}")

    if save_plots:
        plot_compare_runs(
            problem,
            x_histories,
            ec_histories,
            labels=labels,
            save_path=f"{problem.name}/core/grid_search.png",
            obj_lim=obj_lim,
            ec_lim=ec_lim,
        )
        plot_compare_stds(
            problem,
            x_histories_list,
            ec_histories_list,
            labels=labels,
            save_path=f"{problem.name}/core/grid_search_stds.png",
            obj_lim=obj_lim,
            ec_lim=ec_lim,
        )
    if save_results and repeat > 1:
        save_path = get_save_path(problem, content="grid_search_stats", repeat=repeat)
        df_stats.T.drop(columns=["x_mean", "x_std"], errors="ignore").to_csv(get_file_path(save_path, base_dir=base_dir))


def run_core(
    problem: Problem,
    ec_samples_num: int = 100000,
    verbose: bool = True,
    save_path: str = None,
) -> tuple[np.ndarray, np.ndarray, float]:

    optimizer = Optimizer(
        problem=problem,
        zero_order_method=False,
        empirical_coverage=True,
        verbose=verbose,
    )

    start_time = time()
    result = optimizer.run()
    end_time = time()
    run_time = end_time - start_time

    if result is None:
        raise ValueError("result is None")

    if verbose:
        final_ec = Oracle.empirical_coverage(problem, result, N=ec_samples_num)
        final_f_value = float(problem.f_function(result))

        print(f"\nFinal solution: {result}")
        print(f"Objective value: {final_f_value:.6f}")
        print(f"Empirical coverage: {final_ec:.4f}")
        print(f"Solver time: {run_time:.4f} seconds")

    x_history = optimizer.x_history
    ec_history = Oracle.empirical_coverage_history(problem, x_history, N=ec_samples_num)


    if save_path:
        history = {
            "x": [np.asarray(x).tolist() for x in x_history],
            "ec": [float(ec) for ec in ec_history],
        }
        file_path = get_file_path(save_path, base_dir)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return x_history, ec_history, run_time


def run_taco(
    core_problem: Problem, ec_samples_num: int = 100000, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    taco_problem = ToyProblem(problem=core_problem)

    pb_numba_compliant = True

    other_selected_params = {
        "nb_iterations": core_problem.max_iter_x,  # number of iterations
        "bund_mu_high": 0.001,  # upper bound for the proximal parameter of the bundle
        "bund_mu_low": 0.001,  # lower bound for the proximal parameter of the bundle
        "bund_max_size_bundle_set": 31,  # Maximum size for the smoothing constant
        "superquantile_smoothing_param": 0.1,  # smoothing constant
    }

    optimizer = TacoOptimizer(
        problem=taco_problem,
        p=1 - core_problem.delta,
        starting_point=np.concatenate([core_problem.initial_x, [0.0]]),
        numba=pb_numba_compliant,
        params=other_selected_params,
    )

    start_time = time()
    result = optimizer.run(verbose=False)
    end_time = time()
    run_time = end_time - start_time

    if verbose:
        final_ec = Oracle.empirical_coverage(
            core_problem, result[:-1], N=ec_samples_num
        )
        f_value = float(core_problem.f_function(result[:-1]))

        print(f"\nFinal solution: {result[:-1]}")
        print(f"Objective value: {f_value:.6f}")
        print(f"Empirical coverage: {final_ec:.4f}")
        print(f"Solver time: {run_time:.4f} seconds")

    x_history = np.array(optimizer.algorithm.x_history)
    ec_history = Oracle.empirical_coverage_history(
        core_problem, x_history, N=ec_samples_num
    )
    return x_history, ec_history, run_time


def run_cpp(
    problem: Problem,
    method: str = "CPP-MIP",
    ec_samples_num: int = 100000,
    verbose: bool = True,
    save_path: str = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    J = problem.cpp_objective_function
    J_value = problem.f_function
    f = problem.cpp_chance_function
    f_value = problem.chance_function
    x_dim = problem.dimension[0]
    delta = problem.delta
    gs = []
    hs = []
    training_Ys = problem.z_samples()
    result, solver_time = solve(x_dim, delta, training_Ys, hs, gs, f, J, method)

    # print("cpp result =")
    # print(result)
    if result == "timelimit":
        if verbose:
            print("\nSolver hit the time limit.")
        return result, None, solver_time
    final_ec = Oracle.empirical_coverage(problem, result, N=ec_samples_num)
    result = np.array([np.array(result)])

    if verbose:
        _f_value = J_value(result)

        print(f"\nFinal solution: {result[-1]}")
        print(f"Objective value: {_f_value}")
        print(f"Empirical coverage: {final_ec:.4f}")
        print(f"Solver time: {solver_time:.4f} seconds")

    return result, np.array([final_ec]), solver_time


def pad_arr(arr: np.ndarray, target_len: int) -> np.ndarray:
    pad_len = target_len - arr.shape[0]
    if pad_len <= 0:
        return arr
    if arr.ndim == 1:
        return np.pad(arr, (0, pad_len), mode="edge")
    else:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[0] = (0, pad_len)
        return np.pad(arr, pad_width, mode="edge")


def pad_histories(histories: list[np.ndarray]) -> np.ndarray:
    max_len = max(history.shape[0] for history in histories)
    histories_padded = [pad_arr(x_history, max_len) for x_history in histories]
    return np.array(histories_padded)


def _run_single_algorithm_iteration(args):
    problem, method, ec_samples_num, verbose, run_index = args
    # Set different random seed for each process to ensure different results
    # np.random.seed(None)  # Use system time/entropy for seeding
    np.random.seed(mp.current_process().pid + int(time()))
    if verbose:
        print("*" * 20)
        print(f"\nMethod={method}, Run {run_index+1}")
        
    if method == "Core":
        x_history, ec_history, run_time = run_core(
            problem, ec_samples_num=ec_samples_num, verbose=verbose
        )
    elif method == "TACO":
        x_history, ec_history, run_time = run_taco(
            problem, ec_samples_num=ec_samples_num, verbose=verbose
        )
    elif method in ["CPP-MIP", "CPP-KKT", "SA", "SAA"]:
        j = 0
        while j < 3:
            x_history, ec_history, run_time = run_cpp(
                problem,
                method=method,
                ec_samples_num=ec_samples_num,
                verbose=verbose,
            )
            if isinstance(x_history, str) and x_history == "timelimit":
                if verbose and j < 2:
                    print(f"Run {run_index+1} hit the time limit. Repeating...")
                if verbose and j == 2:
                    print(f"Run {run_index+1} hit the time limit 3 times. Skipping...")
                j += 1
            else:
                break
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return x_history, ec_history, run_time, run_index


def run_algorithm(
    problem: Problem,
    method: str = "Core",
    repeat: int = 1,
    ec_samples_num: int = 100000,
    save_results: bool = False,
    verbose: bool = True,
    parallel: bool = False,
    n_jobs: int = None,
) -> dict:
    np.random.seed()

    x_histories = []
    ec_histories = []
    run_times = []
    
    if verbose:
        print(f"Running {method} Algorithm for {repeat} repetitions.")
    
    if parallel and repeat > 1:
        # Use parallel execution for multiple runs
        if n_jobs is None:
            n_jobs = min(max(1, cpu_count()//2 - 1), repeat)
        
        if verbose:
            print(f"Using parallel execution with {n_jobs} processes.")
        
        # Prepare arguments for parallel execution
        args_list = [(problem, method, ec_samples_num, False, i) for i in range(repeat)]
        pool = Pool(processes=n_jobs)

        with pool:
            if not verbose:
                # Use tqdm for progress tracking
                results = list(tqdm(
                    pool.imap(_run_single_algorithm_iteration, args_list),
                    total=repeat,
                    desc=f"Running {method}"
                ))
            else:
                results = pool.map(_run_single_algorithm_iteration, args_list)
        
        # Sort results by run_index to maintain order
        # results.sort(key=lambda x: x[3])
        
        for x_history, ec_history, run_time, _ in results:
            x_histories.append(x_history)
            ec_histories.append(ec_history)
            run_times.append(run_time)
    
    else:
        # Use sequential execution (original behavior)
        for i in tqdm(range(repeat), disable=verbose, desc=f"Running {method}"):
            x_history, ec_history, run_time, _ = _run_single_algorithm_iteration((problem, method, ec_samples_num, verbose, i))

            x_histories.append(x_history)
            ec_histories.append(ec_history)
            run_times.append(run_time)

    if save_results:
        save_path = get_save_path(
            problem, content="eval", method=method, repeat=repeat
        )
        results = get_results_from_histories(
            problem=problem,
            method=method,
            x_histories=x_histories,
            ec_histories=ec_histories,
            run_times=run_times,
            save_path=save_path,
        )
        save_path = get_save_path(
            problem, content="method_stats", repeat=repeat, method=method
        )
        stats = pd.Series(get_stats(results))
        stats.to_csv(get_file_path(save_path, base_dir=base_dir))
        print("=" * 100)
        print(stats)
        print("=" * 100)
    return x_histories, ec_histories, run_times
