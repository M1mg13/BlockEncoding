import itertools
import math
from typing import List, Tuple

import numpy as np
import pytest

# ---------- Fixtures --------------------------------------------------------


@pytest.fixture
def rng():
    """Deterministic RNG for tests (change seed here if needed)."""
    return np.random.default_rng(123456)


# ---------- Basic utilities -------------------------------------------------


def required_qubits(dim: int) -> int:
    if dim <= 1:
        return 0
    return math.ceil(math.log2(dim))


def assert_unitary(U: np.ndarray, atol: float = 1e-10) -> None:
    U = np.asarray(U, dtype=complex)
    assert U.ndim == 2 and U.shape[0] == U.shape[1]
    err = np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]), ord=np.inf)
    assert err <= atol, f"Matrix not unitary (||U^†U - I||_inf = {err})"


def pad_unitary_to_k(U_mat: np.ndarray, k: int) -> np.ndarray:
    """Pad a (m x m) unitary to size k x k with identity block."""
    U_mat = np.asarray(U_mat, dtype=complex)
    n = U_mat.shape[0]
    return np.block(
        [
            [U_mat, np.zeros((n, k - n), dtype=complex)],
            [np.zeros((k - n, n), dtype=complex), np.eye(k - n, dtype=complex)],
        ]
    )


# ---------- MPO & tensor-network helpers -----------------------------------


def brute_contract_mpo(
    Ms: List[np.ndarray], lc: np.ndarray, rc: np.ndarray
) -> np.ndarray:
    """
    Contract an open-chain MPO into a dense matrix.
    Ms: list of tensors shaped (Dl, Dr, p, p)
    lc: left vector (Dl_first,)
    rc: right vector (Dr_last,)
    Returns: dense matrix shape (p^n, p^n) with LSB ordering per-site.
    """
    n = len(Ms)
    if n == 0:
        return np.zeros((1, 1), dtype=complex)

    # validation
    p = Ms[0].shape[2]
    for s, M in enumerate(Ms):
        assert M.ndim == 4, f"MPO site {s} wrong ndim"
        assert M.shape[2] == p and M.shape[3] == p, "physical dims mismatch"
        if s > 0:
            assert Ms[s - 1].shape[1] == M.shape[0], "bond dims mismatch"

    lc = np.asarray(lc, dtype=complex)
    rc = np.asarray(rc, dtype=complex)
    assert lc.shape[0] == Ms[0].shape[0]
    assert rc.shape[0] == Ms[-1].shape[1]

    # fold left boundary
    T = np.tensordot(lc, Ms[0], axes=([0], [0]))  # (Dr, p, p)

    for s in range(1, n):
        # contract next site on left bond -> new T with expanded physical legs
        T = np.tensordot(Ms[s], T, axes=([0], [0]))
        # bring dims to (Dr, p, p^(s), p^(s)) ordering in LSB sense and flatten
        T = T.transpose(0, 1, 3, 2, 4).reshape(-1, p ** (s + 1), p ** (s + 1))

    H = np.tensordot(rc, T, axes=([0], [0]))  # shape (p^n, p^n)
    return H


def random_mpo_cores(n: int, bond_dim: int, rng=None, p: int = 2) -> List[np.ndarray]:
    """
    Build MPO cores using provided rng. rng may be:
      - a numpy.random.Generator instance (preferred, e.g. the pytest 'rng' fixture)
      - an int seed (will create a Generator)
      - None (uses a default deterministic seed for legacy behavior)
    """
    if isinstance(rng, int) or rng is None:
        rng = np.random.default_rng(123 if rng is None else rng)
    # if user passed a Generator already, use it directly
    cores = []
    for i in range(n):
        Dl = 1 if i == 0 else bond_dim
        Dr = 1 if i == n - 1 else bond_dim
        core = rng.normal(size=(Dl, Dr, p, p)) + 1j * rng.normal(size=(Dl, Dr, p, p))
        cores.append(core.astype(complex))
    return cores


# ---------- QUBO helper ----------------------------------------------------


def qubo_direct_dense(Q: np.ndarray, c: float = 0.0) -> np.ndarray:
    """
    Build diagonal dense H for QUBO E(x) = x^T Q x + c.
    Bit ordering: LSB rightmost; x[i] == bit i of integer b (i from 0..n-1).
    Returns (2^n x 2^n) ndarray (diagonal).
    """
    Q = np.asarray(Q, dtype=complex)
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square")
    N = 1 << n
    H = np.zeros((N, N), dtype=complex)
    for b in range(N):
        x = np.array([(b >> i) & 1 for i in range(n)], dtype=float)
        H[b, b] = c + x @ Q @ x
    return H


# ---------- compare / permutation helpers ----------------------------------


def permute_qubits_matrix(H: np.ndarray, perm: Tuple[int, ...]) -> np.ndarray:
    """
    Permute qubits (sites) of a 2^n x 2^n matrix. perm is tuple of length n
    mapping new ordering: output qubit i is original perm[i].
    """
    H = np.asarray(H, dtype=complex)
    dim = H.shape[0]
    n = int(math.log2(dim))
    assert H.shape == (1 << n, 1 << n)
    tens = H.reshape((2,) * (2 * n))
    outs = list(range(n))
    ins = list(range(n, 2 * n))
    perm_out = [outs[i] for i in perm]
    perm_in = [ins[i] for i in perm]
    axes = perm_out + perm_in
    tens_perm = tens.transpose(axes)
    return tens_perm.reshape(dim, dim)


def optimal_scalar(A: np.ndarray, B: np.ndarray) -> complex:
    """Find scalar s minimizing ||s B - A||_2 in least-squares sense."""
    a = A.ravel()
    b = B.ravel()
    denom = np.vdot(a, a)
    if denom == 0:
        return 0.0 + 0j
    return np.vdot(a, b) / denom


def find_best_perm_and_scale(A: np.ndarray, B: np.ndarray, max_permute: int = 720):
    """
    Try permutations of qubits (full if feasible) and compute optimal scalar scaling.
    Returns (residual, best_perm, best_scale).
    """
    dim = A.shape[0]
    n = int(math.log2(dim))
    assert A.shape == B.shape
    if math.factorial(n) <= max_permute:
        perms = list(itertools.permutations(range(n)))
    else:
        # small heuristic set
        perms = [tuple(range(n)), tuple(reversed(range(n)))]
        if n >= 2:
            swap = list(range(n))
            swap[0], swap[1] = swap[1], swap[0]
            perms.append(tuple(swap))

    best = (np.inf, None, None)
    for p in perms:
        Bp = permute_qubits_matrix(B, p)
        s = optimal_scalar(Bp, A)
        res = np.linalg.norm(s * Bp - A)
        if res < best[0]:
            best = (res, p, s)
    return best


def compare_matrices_allow_perm_scale(A: np.ndarray, B: np.ndarray, tol: float = 1e-8):
    """Convenience wrapper returning (res, perm, scale)."""
    return find_best_perm_and_scale(A, B)
