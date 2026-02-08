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
        self.objective_grad = lambda x: self.oracle.x_partial_F_function(
            x=x, min_s=self.minimize_s(x=x, initial_s=0), z_samples=self.z_samples
        )

    def run(self) -> np.ndarray:
        # TODO: samples management

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

    def minimize_s(self, x: np.ndarray, initial_s: float):
        initial_s = 0
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
        return final_s

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

            # early stopping if requested (based on parameter change)
            if abstol is not None and np.linalg.norm(x - prev_x) < abstol:
                # if outer:
                #     print(f"\nConverged in {t} iterations.")
                break

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
