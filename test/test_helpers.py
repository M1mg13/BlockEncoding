import itertools

import numpy as np
import pytest

from BlockEncoding import (
    QUBO_to_ising_parts,
    identity_embed_pad,
    identity_pad_to_power,
    pad_tensor,
    unitary_diag_dilation,
    unitary_dilation,
)


def brute_qubo_to_ising_coeffs(Q, c=0.0):
    """
    Brute-force compute coefficients (offset, linear, pairwise) for the expansion
    H(z) = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j where z_i in {-1,1}
    and x = (1 - z)/2 with E(x) = x^T Q x + c.
    Returns (const, linear_list, square_dict).
    """
    n = Q.shape[0]
    zs = list(itertools.product([-1.0, 1.0], repeat=n))
    vals = []
    for z in zs:
        x = np.array([(1 - zi) / 2.0 for zi in z])
        vals.append(float(c + x @ Q @ x))
    vals = np.array(vals)

    # Build regression matrix for terms [1, z1..zn, z1*z2..]
    cols = [np.ones(len(zs))]
    for i in range(n):
        cols.append(np.array([z[i] for z in zs]))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            cols.append(np.array([z[i] * z[j] for z in zs]))
    A = np.vstack(cols).T
    coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)
    const = float(coeffs[0])
    linear = [float(c) for c in coeffs[1 : 1 + n]]
    square = {p: float(coeffs[1 + n + idx]) for idx, p in enumerate(pairs)}
    return const, linear, square


@pytest.mark.parametrize("n", [1, 2, 3])
def test_QUBO_to_ising_parts_matches_bruteforce(n):
    rng = np.random.default_rng(42)
    A = rng.normal(scale=0.5, size=(n, n))
    Q = 0.5 * (A + A.T)
    c = 0.17

    const, linear, square = QUBO_to_ising_parts(Q, c=c, tol=1e-12)
    bf_const, bf_linear, bf_square = brute_qubo_to_ising_coeffs(Q, c=c)

    assert np.allclose(const, bf_const, atol=1e-12, rtol=0)
    assert np.allclose(
        np.array(linear, dtype=complex),
        np.array(bf_linear, dtype=complex),
        atol=1e-12,
        rtol=0,
    )

    for i in range(n):
        for j in range(i + 1, n):
            a = square.get((i, j), 0.0)
            b = bf_square.get((i, j), 0.0)
            assert np.allclose(a, b, atol=1e-12, rtol=0)


def test_QUBO_to_ising_parts_simple_examples():
    Q = np.array([[2.0]])
    const, linear, square = QUBO_to_ising_parts(Q, c=0.5, tol=1e-12)
    assert np.allclose(const, 1.5)
    assert np.allclose(linear, [-1.0])
    assert square == {}

    Q = np.array([[0.0, 4.0], [0.0, 0.0]])
    const, linear, square = QUBO_to_ising_parts(Q, c=0.0, tol=1e-12)
    assert np.allclose(const, 1.0)
    assert np.allclose(linear, [-1.0, -1.0])
    assert np.allclose(square.get((0, 1), 0.0), 1.0)


def test_unitary_diag_dilation_edge_cases():
    s, c, g = unitary_diag_dilation([], gamma=None, tol=1e-12)
    assert s.size == 0 and c.size == 0
    assert np.isclose(g, 1.0)

    s_in = np.array([0.4, 0.2])
    s2, c2, g2 = unitary_diag_dilation(s_in, gamma=0.0, tol=1e-8)
    assert np.allclose(s2, np.zeros_like(s_in))
    assert np.allclose(c2, np.ones_like(s_in))
    assert g2 == 0.0

    with pytest.raises(ValueError):
        unitary_diag_dilation([1.0, 0.8], gamma=0.5, tol=1e-12)


@pytest.mark.parametrize("method", ["square", "rect"])
def test_unitary_dilation_reconstructs_top_left_block(method):
    rng = np.random.default_rng(7)
    M = rng.normal(size=(2, 2)) + 0.0j
    Q, alpha = unitary_dilation(M, method=method, tol=1e-12)
    assert np.allclose(Q.conj().T @ Q, np.eye(Q.shape[0]), atol=1e-10)
    k = max(*M.shape)
    A_block = Q[:k, :k]
    assert np.allclose(alpha * A_block[: M.shape[0], : M.shape[1]], M, atol=1e-10)


def test_pad_and_identity_embed_pad_behavior():
    M = np.array([[1, 2], [3, 4]], dtype=complex)
    P = pad_tensor(M, (3, 4))
    assert P.shape == (3, 4)
    assert P[0, 0] == 1 and P[1, 0] == 3 and P[1, 1] == 4
    assert np.allclose(P[2, :], 0)

    M2 = np.zeros((2, 3), dtype=complex)
    M2[0, 0] = 2.0 + 0j
    M2[1, 1] = 3.0 + 0j
    dims = (4, 5)
    T = identity_embed_pad(M2, dims, out_axes=(0,), in_axes=(1,))
    T_unfold = T.transpose(0, 1).reshape(4, 5)
    assert np.allclose(T_unfold[0, 0], 2.0)
    assert np.allclose(T_unfold[1, 1], 3.0)
    assert np.allclose(T_unfold[2, 2], 1.0)
    assert np.allclose(T_unfold[3, 3], 1.0)

    M3 = np.zeros((3, 3), dtype=complex)
    P3 = identity_pad_to_power(M3, base=2, out_axes=(0,), in_axes=(1,))
    assert P3.shape == (4, 4)
