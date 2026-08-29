"""轻量元启发式优化器（纯 numpy，无外部依赖）。
- pso: 粒子群，用于 SVR 超参 (C, gamma, epsilon) 在 log 空间搜索。
- ga:  实数编码遗传算法，用于 ANN 超参 (hidden, alpha, lr) 搜索。
均最小化目标 f(x)。"""
import numpy as np


def pso(f, bounds, n_particles=14, iters=24, seed=0, w=0.7, c1=1.6, c2=1.6, verbose=False):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    X = lb + rng.random((n_particles, dim)) * (ub - lb)
    V = (rng.random((n_particles, dim)) - 0.5) * (ub - lb)
    pbest = X.copy()
    pbest_val = np.array([f(x) for x in X])
    gidx = int(np.argmin(pbest_val))
    g = X[gidx].copy()
    gval = float(pbest_val[gidx])
    for it in range(iters):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (g - X)
        X = np.clip(X + V, lb, ub)
        val = np.array([f(x) for x in X])
        imp = val < pbest_val
        pbest[imp] = X[imp]
        pbest_val[imp] = val[imp]
        bi = int(np.argmin(pbest_val))
        if pbest_val[bi] < gval:
            gval = float(pbest_val[bi])
            g = pbest[bi].copy()
        if verbose:
            print(f"    [PSO it {it+1}/{iters}] best={gval:.4f}")
    return g, gval


def ga(f, bounds, pop=16, gens=20, seed=0, mut_rate=0.25, mut_scale=0.15, verbose=False):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    span = ub - lb
    P = lb + rng.random((pop, dim)) * span
    val = np.array([f(x) for x in P])
    for gen in range(gens):
        order = np.argsort(val)
        n_parents = max(2, pop // 2)
        parents = order[:n_parents]
        children = [P[order[0]].copy()]  # elitism: keep best
        while len(children) < pop:
            a, b = rng.choice(parents, 2, replace=True)
            alpha = rng.random()
            c = alpha * P[a] + (1 - alpha) * P[b]
            if rng.random() < mut_rate:
                c = c + rng.normal(0, mut_scale) * span
            children.append(np.clip(c, lb, ub))
        P = np.array(children[:pop])
        val = np.array([f(x) for x in P])
        if verbose:
            print(f"    [GA gen {gen+1}/{gens}] best={val.min():.4f}")
    bi = int(np.argmin(val))
    return P[bi], float(val[bi])
