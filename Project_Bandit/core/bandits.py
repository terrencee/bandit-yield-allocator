import numpy as np

def epsilon_greedy(R, eps=0.05, switch_cost=0.0001, seed=0):
    rng = np.random.default_rng(seed)
    T, K = R.shape
    counts = np.zeros(K, int); values = np.zeros(K, float)
    picks  = np.zeros(T, int); rew = np.zeros(T, float)
    for t in range(T):
        if t < K: a = t
        else:     a = rng.integers(K) if rng.random() < eps else int(np.argmax(values))
        r_t = R[t, a]
        if t > 0 and a != picks[t-1]: r_t -= switch_cost
        counts[a] += 1
        values[a] += (r_t - values[a]) / counts[a]
        picks[t], rew[t] = a, r_t
    return picks, rew

def linucb_disjoint(X, R, alpha=0.5):
    T, d = X.shape; K = R.shape[1]
    A = [np.eye(d) for _ in range(K)]
    b = [np.zeros(d)  for _ in range(K)]
    picks = np.zeros(T, int); rew = np.zeros(T, float)
    for t in range(T):
        x = X[t]; p = np.zeros(K)
        for a in range(K):
            Ainv = np.linalg.inv(A[a]); theta = Ainv @ b[a]
            p[a] = theta @ x + alpha * np.sqrt(x @ Ainv @ x)
        a = int(np.argmax(p)); r_t = R[t, a]
        A[a] += np.outer(x, x); b[a] += r_t * x
        picks[t], rew[t] = a, r_t
    return picks, rew
