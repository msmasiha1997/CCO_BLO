from __future__ import annotations

import sys
import os
import torch
import numpy as np
from ..problems.problem import Problem
from .oracle import Oracle
from tqdm import tqdm


class Optimizer:
    def __init__(
        self,
        problem: Problem,
        zero_order_method: bool,
        empirical_coverage: bool = False,
        verbose: bool = True,
        abstol: float | None = 1e-9,
    ) -> None:

        self.problem = problem
        self.oracle = Oracle(problem)
        self.x = self.problem.initial_x
        self.x_history = None
        self.objective_function_history = np.zeros(self.problem.max_iter_x)
        self.chance_constraint_quantile_history = np.zeros(self.problem.max_iter_x)
        self.empirical_coverage_history = np.zeros(self.problem.max_iter_x)
        self.feasible_x = []
        self.zero_order_method = zero_order_method
        self.empirical_coverage = empirical_coverage
        self.verbose = verbose
        self.zo_double_evaluation = False if problem.dimension == (1,) else True

        self.z_samples = self.problem.z_samples()
        self.inner_solver = str(getattr(self.problem, "inner_solver", "bisect"))
        self.inner_tol = float(getattr(self.problem, "inner_tol", 1e-8))
        self.mu_schedule = str(getattr(self.problem, "mu_schedule", "constant"))
        self.mu_final = getattr(self.problem, "mu_final", None)
        self.feasible_selection = bool(getattr(self.problem, "feasible_selection", True))
        self.feasible_repair = bool(getattr(self.problem, "feasible_repair", True))
        self.feasible_tol = float(getattr(self.problem, "feasible_tol", 0.0))
        self.repair_steps = int(getattr(self.problem, "repair_steps", 25))
        target_default = 1.0 - float(self.problem.delta)
        self.feasible_target = float(getattr(self.problem, "feasible_target", target_default))
        self.feasible_target = float(np.clip(self.feasible_target, 0.0, 1.0))

        self._last_s: float = 0.0
        self._grad_calls: int = 0
        self.objective_grad = self._objective_grad

    def run(self) -> np.ndarray:
        # TODO: samples management
        self._last_s = 0.0
        self._grad_calls = 0

        final_x, x_history = self.Adam(
            initial_x=self.problem.initial_x,
            grad_func=self.objective_grad,
            lr=self.problem.lr,
            max_iter=self.problem.max_iter_x,
            abstol=self.problem.abstol,
            grad_clip=self.problem.grad_clip,
            update_clip=self.problem.update_clip,
            history=True,
            enable_tqdm=self.verbose, #TODO: everything is fine?
            use_torch=False,
            outer=True,
        )

        self.x_history = x_history
        return final_x

    def _mu_at(self, step: int) -> float:
        mu0 = float(self.problem.mu)
        if self.mu_final is None or self.mu_schedule == "constant" or self.problem.max_iter_x <= 1:
            return mu0
        muf = float(self.mu_final)
        if self.mu_schedule == "linear":
            t = step / max(1, self.problem.max_iter_x - 1)
            return float(mu0 + t * (muf - mu0))
        if self.mu_schedule == "geom":
            if mu0 <= 0 or muf <= 0:
                return mu0
            t = step / max(1, self.problem.max_iter_x - 1)
            return float(mu0 * ((muf / mu0) ** t))
        return mu0

    def _objective_grad(self, x: np.ndarray) -> np.ndarray:
        min_s = float(self.minimize_s(x=x, initial_s=self._last_s))
        self._last_s = min_s
        mu_k = self._mu_at(self._grad_calls)
        self._grad_calls += 1
        return self.oracle.x_partial_F_function(
            x=x,
            min_s=min_s,
            z_samples=self.z_samples,
            mu=mu_k,
        )

    def _train_ec(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=float).reshape(self.problem.dimension)
        chance_values = np.asarray(
            self.problem.chance_function(x_arr, self.z_samples), dtype=float
        ).reshape(-1)
        if chance_values.size == 0:
            return 0.0
        return float(np.mean(chance_values <= 0.0))

    def _is_feasible_ec(self, ec: float) -> bool:
        return bool(ec >= (self.feasible_target - self.feasible_tol))

    def _repair_to_feasible(self, x_infeasible: np.ndarray, x_feasible: np.ndarray) -> np.ndarray:
        x_bad = np.asarray(x_infeasible, dtype=float).reshape(self.problem.dimension)
        x_safe = np.asarray(x_feasible, dtype=float).reshape(self.problem.dimension)
        if self._is_feasible_ec(self._train_ec(x_bad)):
            return x_bad
        if not self._is_feasible_ec(self._train_ec(x_safe)):
            return x_safe

        lo = 0.0
        hi = 1.0
        grid = np.linspace(0.0, 1.0, num=11)
        for i in range(1, len(grid)):
            alpha = float(grid[i])
            cand = (1.0 - alpha) * x_bad + alpha * x_safe
            if self._is_feasible_ec(self._train_ec(cand)):
                hi = alpha
                lo = float(grid[i - 1])
                break

        best = (1.0 - hi) * x_bad + hi * x_safe
        for _ in range(max(5, self.repair_steps)):
            mid = 0.5 * (lo + hi)
            cand = (1.0 - mid) * x_bad + mid * x_safe
            if self._is_feasible_ec(self._train_ec(cand)):
                best = cand
                hi = mid
            else:
                lo = mid

        return best

    def _select_outer_solution(
        self,
        x_final: np.ndarray,
        best_feasible_x: np.ndarray | None,
        best_feasible_obj: float,
        repair_anchor_x: np.ndarray | None,
    ) -> np.ndarray:
        x_selected = np.asarray(x_final, dtype=float).reshape(self.problem.dimension)
        candidates: list[tuple[float, np.ndarray]] = []

        ec_final = self._train_ec(x_selected)
        if self._is_feasible_ec(ec_final):
            candidates.append((float(self.problem.f_function(x_selected)), x_selected.copy()))

        if best_feasible_x is not None and np.isfinite(best_feasible_obj):
            candidates.append((float(best_feasible_obj), best_feasible_x.copy()))

        if (
            self.feasible_repair
            and repair_anchor_x is not None
            and not self._is_feasible_ec(ec_final)
        ):
            repaired = self._repair_to_feasible(x_selected, repair_anchor_x)
            ec_repaired = self._train_ec(repaired)
            if self._is_feasible_ec(ec_repaired):
                candidates.append((float(self.problem.f_function(repaired)), repaired.copy()))

        if not candidates:
            return x_selected
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1]

    def minimize_s(self, x: np.ndarray, initial_s: float):
        if self.inner_solver == "bisect":
            return self._minimize_s_bisect(x=x, initial_s=initial_s)
        final_s, _ = self.Adam(
            initial_x=initial_s,
            grad_func=lambda s: self.oracle.s_partial_G_function(
                x, s, z_samples=self.z_samples
            ),
            lr=self.problem.lr,
            max_iter=self.problem.max_iter_s,
            abstol=self.problem.abstol,
            grad_clip=self.problem.grad_clip,
            update_clip=self.problem.update_clip,
            enable_tqdm=False,
            history=False,
            use_torch=False,
            outer=False,
        )
        return float(np.asarray(final_s).reshape(-1)[0])

    def _minimize_s_bisect(self, x: np.ndarray, initial_s: float) -> float:
        gvals = np.asarray(self.problem.chance_function(x, self.z_samples), dtype=float).reshape(-1)
        if gvals.size == 0:
            return float(initial_s)

        def dG_ds(s: float) -> float:
            chance_values = gvals - float(s)
            expected_estimation = np.mean(self.problem.partial_h_function(chance_values))
            return float(1.0 - expected_estimation / float(self.problem.delta) + float(self.problem.epsilon) * float(s))

        span = max(1.0, float(np.max(np.abs(gvals))), abs(float(initial_s)))
        lo = float(np.min(gvals) - span)
        hi = float(np.max(gvals) + span)
        d_lo = dG_ds(lo)
        d_hi = dG_ds(hi)

        for _ in range(60):
            if d_lo <= 0.0 and d_hi >= 0.0:
                break
            width = max(1.0, hi - lo)
            if d_lo > 0.0:
                lo -= 2.0 * width
                d_lo = dG_ds(lo)
            if d_hi < 0.0:
                hi += 2.0 * width
                d_hi = dG_ds(hi)
        else:
            return float(initial_s)

        iters = max(20, int(self.problem.max_iter_s))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            d_mid = dG_ds(mid)
            if d_mid <= 0.0:
                lo = mid
            else:
                hi = mid
            if (hi - lo) <= self.inner_tol:
                break

        return float(0.5 * (lo + hi))

    def torch_SGD(
        self,
        s_0: float,
        grad_func,
        history: bool = False,
        device: str = "cpu",
        dtype=None,
        abstol: float | None = 1e-3,
    ) -> tuple[np.ndarray | float, list | None]:

        if dtype is None:
            dtype = torch.float64
        device = torch.device(device)

        s_param = torch.nn.Parameter(torch.tensor(s_0, dtype=dtype, device=device))
        opt = torch.optim.SGD([s_param], lr=self.problem.lr)

        s_history = [float(s_param.detach().cpu().numpy())] if history else None
        prev_val = float(s_param.detach().cpu().numpy())

        for t in range(self.problem.max_iter_s):
            s_numpy = s_param.detach().cpu().numpy()
            g = grad_func(s_numpy)

            g_t = torch.as_tensor(g, dtype=dtype, device=device)

            opt.zero_grad()
            s_param.grad = g_t

            opt.step()

            if history:
                s_history.append(float(s_param.detach().cpu().numpy()))

            if abstol is not None:
                curr = float(s_param.detach().cpu().numpy())
                if abs(curr - prev_val) < abstol:
                    break
                prev_val = curr

        final = s_param.detach().cpu().numpy()
        return (final, s_history) if history else (final, None)

    def custom_SGD(
        self,
        s_0: float,
        grad_func,
        history: bool = False,
        abstol: float = 1e-3,
    ) -> tuple[np.ndarray | float, list | None]:
        s_t = s_0

        if history:
            s_history = [s_0]

        for i in range(self.problem.max_iter_s):
            prev_s = s_t
            gradient = grad_func(s_t)
            s_t -= self.problem.lr * gradient

            if history:
                s_history.append(s_t)

            # check for early convergence
            if abstol is not None and np.linalg.norm(s_t - prev_s) < abstol:
                break

        return (s_t, s_history) if history else (s_t, None)

    def SGD(
        self,
        s_0: float,
        grad_func,
        history: bool = False,
        device: str = "cpu",
        dtype=None,
        abstol: float | None = 1e-3,
        use_torch: bool = False,
    ) -> tuple[np.ndarray | float, list | None]:

        if use_torch:
            return self.torch_SGD(
                s_0=s_0,
                grad_func=grad_func,
                history=history,
                device=device,
                dtype=dtype,
                abstol=abstol,
            )

        return self.custom_SGD(
            s_0=s_0, grad_func=grad_func, history=history, abstol=abstol
        )

    def custom_Adam(
        self,
        initial_x: np.ndarray,
        grad_func: callable,
        lr: float,
        max_iter: int,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-7,
        abstol: float | None = 1e-3,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        history: bool = False,
        enable_tqdm: bool = False,
        outer: bool = False,
    ) -> tuple[np.ndarray | float, list | None]:
        beta1, beta2 = betas

        x = (
            initial_x.copy()
            if isinstance(initial_x, np.ndarray)
            else np.array([initial_x])
        )
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        if history:
            x_history = [x]

        best_feasible_x = None
        best_feasible_obj = float("inf")
        repair_anchor_x = None
        if outer and self.feasible_selection:
            ec0 = self._train_ec(x)
            if self._is_feasible_ec(ec0):
                best_feasible_x = x.copy()
                best_feasible_obj = float(self.problem.f_function(x))
                repair_anchor_x = x.copy()

        for t in tqdm(range(1, max_iter + 1), disable=not enable_tqdm):
            prev_x = x.copy()
            gradient = grad_func(x)

            # gradient clipping (clip by global norm)
            if grad_clip is not None:
                g_norm = np.linalg.norm(gradient)
                if g_norm > 0 and g_norm > grad_clip:
                    gradient = gradient * (grad_clip / g_norm)


            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)


            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # compute parameter update
            update = lr * m_hat / (np.sqrt(v_hat) + epsilon)

            # optional update clipping (limit step norm)
            if update_clip is not None:
                u_norm = np.linalg.norm(update)
                if u_norm > 0 and u_norm > update_clip:
                    update = update * (update_clip / u_norm)

            # apply update
            x = x - update

            if history:
                x_history.append(x)

            if outer and self.feasible_selection:
                ec_x = self._train_ec(x)
                if self._is_feasible_ec(ec_x):
                    obj_x = float(self.problem.f_function(x))
                    if obj_x < best_feasible_obj:
                        best_feasible_obj = obj_x
                        best_feasible_x = x.copy()
                    repair_anchor_x = x.copy()

            # early stopping if requested (based on parameter change)
            if abstol is not None and np.linalg.norm(x - prev_x) < abstol:
                # if outer:
                #     print(f"\nConverged in {t} iterations.")
                break

        if outer and self.feasible_selection:
            selected_x = self._select_outer_solution(
                x_final=x,
                best_feasible_x=best_feasible_x,
                best_feasible_obj=best_feasible_obj,
                repair_anchor_x=repair_anchor_x,
            )
            if history and np.linalg.norm(selected_x - x) > 0:
                x_history.append(selected_x.copy())
            x = selected_x

        return (x, np.array(x_history)) if history else (x, None)

    def torch_Adam(
        self,
        initial_x: np.ndarray,
        grad_func: callable,
        lr: float,
        max_iter: int,
        device: str = "cpu",
        dtype=None,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-7,
        abstol: float | None = 1e-3,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        history: bool = False,
        enable_tqdm: bool = False,
        outer: bool = False,
    ) -> tuple[np.ndarray | float, list | None]:

        if dtype is None:
            dtype = torch.float64
        device = torch.device(device)

        x_param = torch.nn.Parameter(
            torch.tensor(initial_x, dtype=dtype, device=device)
        )
        opt = torch.optim.Adam([x_param], lr=lr, betas=betas, eps=epsilon)

        x_history = [x_param.detach().cpu().numpy().copy()] if history else None

        prev = x_param.detach().cpu().numpy().copy()

        for t in tqdm(range(max_iter), disable=not enable_tqdm):
            x_numpy = x_param.detach().cpu().numpy()
            g = grad_func(x_numpy)

            g_t = torch.as_tensor(g, dtype=dtype, device=device)

            opt.zero_grad()
            x_param.grad = g_t

            if grad_clip is not None:
                grad_norm = torch.norm(x_param.grad)
                if grad_norm > grad_clip:
                    x_param.grad = x_param.grad * (grad_clip / grad_norm)

            opt.step()

            if history:
                x_history.append(x_param.detach().cpu().numpy().copy())

            if abstol is not None:
                curr = x_param.detach().cpu().numpy()
                if np.linalg.norm(curr - prev) < abstol:
                    break
                prev = curr.copy()

            if update_clip is not None:
                curr = x_param.detach().cpu().numpy()
                u_norm = np.linalg.norm(curr - prev)
                if u_norm > 0 and u_norm > update_clip:
                    x_param.data = (curr - prev) * (update_clip / u_norm) + prev

        final = x_param.detach().cpu().numpy()
        return (final, np.array(x_history)) if history else (final, None)

    def Adam(
        self,
        initial_x: np.ndarray,
        grad_func: callable,
        lr: float,
        max_iter: int,
        device: str = "cpu",
        dtype=None,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-7,
        abstol: float | None = 1e-3,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        history: bool = False,
        enable_tqdm: bool = False,
        use_torch: bool = False,
        outer: bool = False,
    ) -> tuple[np.ndarray | float, list | None]:
        if use_torch:
            return self.torch_Adam(
                initial_x,
                grad_func,
                lr=lr,
                max_iter=max_iter,
                history=history,
                device=device,
                dtype=dtype,
                betas=betas,
                epsilon=epsilon,
                abstol=abstol,
                grad_clip=grad_clip,
                update_clip=update_clip,
                enable_tqdm=enable_tqdm,
                outer=outer,
            )
        return self.custom_Adam(
            initial_x,
            grad_func,
            lr=lr,
            max_iter=max_iter,
            history=history,
            betas=betas,
            epsilon=epsilon,
            abstol=abstol,
            grad_clip=grad_clip,
            update_clip=update_clip,
            enable_tqdm=enable_tqdm,
            outer=outer
        )
