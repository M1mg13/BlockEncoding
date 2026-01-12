import numpy as np
import pytest
from conftest import assert_unitary, brute_contract_mpo

from BlockEncoding import (
    TensorNetworkEncoding,
    base_count,
    identity_pad_to_power,
    next_pow,
    unfold_tensor,
    unitary_diag_dilation,
)


def test_base_count_nonbinary_and_next_pow():
    # base != 2 branch exercised
    assert base_count(9, base=3) == 2
    assert next_pow(9, base=3) == 9
    assert next_pow(10, base=3) == 27


def test_unitary_diag_dilation_alpha_zero_branch():
    s = np.array([1e-14, 1e-13])
    s_pad, c_pad, alpha = unitary_diag_dilation(s, tol=1e-12)
    assert alpha == 0.0
    assert np.allclose(s_pad, 0.0)
    assert np.allclose(c_pad, 1.0)


@pytest.mark.parametrize(
    "a,b,n_sites",
    [
        (2, 1, 1),
        (2, 1, 3),
        (2, 2, 3),
        (2, 3, 2),
        (3, 1, 2),
        (3, 2, 2),
    ],
)
def test_tensor_network_build_dimension_powers_brute_contract(rng, a, b, n_sites):
    p = a**b  # atomic dimension to the power of digits
    phys_total = p**n_sites
    # TensorNetworkEncoding expects tensors with last two axes being (phys_out, phys_in)
    # and earlier axes are bond dims. Single bond per site: shape (1,1,p,p)
    V = [
        (rng.normal(size=(1, 1, p, p)) + 1j * rng.normal(size=(1, 1, p, p))).astype(
            complex
        )
        for _ in range(n_sites)
    ]
    V_brute = [v.copy() for v in V]

    E = -np.ones((n_sites, n_sites), dtype=int)  # No loops and not couplings
    if n_sites == 1:
        V[0] = V[0][0, 0]
    elif n_sites == 2:
        E[0, 1] = 0
        E[1, 0] = 0
        V[0] = V[0][0]
        V[1] = V[1][0]
    else:
        E[n_sites - 1, 0] = 1
        E[0, n_sites - 1] = 0
        for i in range(n_sites):
            E[i, (i + 1) % n_sites] = 1
            E[(i + 1) % n_sites, i] = 0

    T = TensorNetworkEncoding(V, E, pad_pow=a)
    G, gamma = T.build_dense_matrix(max_dims=1 << 10, tol=1e-12)

    assert_unitary(G, atol=1e-9)

    H = brute_contract_mpo(V_brute, np.array([1.0]), np.array([1.0]))

    # Only the top-left physical block is the encoded MPO operator
    assert np.allclose(
        gamma * G[:phys_total, :phys_total], H, atol=1e-9
    ), "Upper-left block mismatch"


# Tests for identity padding behavior on diagonal matrices on both pad_pow and pad_pow=None


@pytest.mark.parametrize("pad_pow", [2, None])
@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3)])
def test_identity_padding_on_diagonal_matrices(rng, pad_pow, shape):
    m, n = shape
    p = min(m, n)
    # diagonal matrix with random diagonal entries
    M = np.zeros((m, n), dtype=complex)
    vals = rng.uniform(0.1, 1.0, size=p)
    for i in range(p):
        M[i, i] = vals[i]

    # If pad_pow is not None, identity_pad_to_power should embed ones on the unfolded diagonal
    if pad_pow is not None:
        T_padded = identity_pad_to_power(M, base=pad_pow, out_axes=(0,), in_axes=(1,))
        M_unfold = unfold_tensor(T_padded, out_axes=(0,), in_axes=(1,))
        diag_old = p
        idx = np.arange(diag_old, min(M_unfold.shape))
        if idx.size > 0:
            assert np.allclose(np.diag(M_unfold)[idx], 1.0 + 0.0j)
        # shape should be powers of pad_pow
        assert T_padded.shape[0] == next_pow(m, pad_pow)
        assert T_padded.shape[1] == next_pow(n, pad_pow)
    else:
        # pad_pow=None: nothing should be added/padded
        # unfolded matrix equals original
        T_no_pad = M.copy()
        assert T_no_pad.shape == (m, n)
        M_unfold = unfold_tensor(T_no_pad, out_axes=(0,), in_axes=(1,))
        # Ensure diagonal entries beyond original size do not exist
        assert M_unfold.shape == (m, n)
        # original diagonal preserved
        assert np.allclose(np.diag(M_unfold)[:p], vals)


def test_build_dense_matrix_memoryerror_for_small_max_dims(rng):
    p = 2
    A = rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))
    # single-site with no bond axes (just phys_out, phys_in)
    M0 = np.zeros((p, p), dtype=complex)
    M0[:, :] = A
    V = [M0]
    E = -np.ones((1, 1), dtype=int)
    T = TensorNetworkEncoding(V, E, pad_pow=2)
    # force tiny max_dims so that 1 << nq exceeds it
    with pytest.raises(MemoryError):
        T.build_dense_matrix(max_dims=1)


def test_Singletons():
    Singletons = []
    p = 2

    for j in range(p):
        for i in range(p):
            M = np.zeros((p, p), dtype=complex)
            M[j, i] = 1
            Singletons.append(M)

    for A in Singletons:
        for B in Singletons:
            T = TensorNetworkEncoding([A[None, ...], B[None, ...]], -np.eye(2))
            G, gamma = T.build_dense_matrix()
            assert_unitary(G)

            H = brute_contract_mpo(
                [A[None, None, ...], B[None, None, ...]],
                np.array([1.0]),
                np.array([1.0]),
            )

            assert np.allclose(
                gamma * G[: p**2, : p**2], H, atol=1e-9
            ), "Upper-left block mismatch"
