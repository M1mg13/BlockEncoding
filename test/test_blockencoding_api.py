import numpy as np
import pytest
from conftest import assert_unitary
from numpy.linalg import svd

from BlockEncoding import (
    TensorNetworkEncoding,
    base_count,
    identity_embed_pad,
    identity_pad_to_power,
    next_pow,
    pad_tensor,
    parse_axes,
    refold_mpo,
    refold_tensor,
    unfold_mpo,
    unfold_tensor,
    unitary_diag_dilation,
    unitary_dilation,
)


def test_base_count_next_pow_basic():
    assert base_count(0) == 0
    assert base_count(1) == 0
    assert base_count(2) == 1
    assert base_count(3) == 2
    assert next_pow(1) == 1
    assert next_pow(2) == 2
    assert next_pow(3) == 4
    assert next_pow(16) == 16


def test_unfold_and_refold_roundtrip(rng):
    T = rng.normal(size=(3, 4, 5)) + 1j * rng.normal(size=(3, 4, 5))
    out_axes = (0, 2)
    in_axes = (1,)
    M = unfold_tensor(T, out_axes, in_axes)
    assert M.shape == (3 * 5, 4)
    T_rec = refold_tensor(M, T.shape, out_axes, in_axes)
    assert np.allclose(T_rec, T)


def test_unfold_mpo_and_refold_mpo(rng):
    Dl, Dr = 3, 5
    M = rng.normal(size=(Dl, Dr, 2, 2)) + 1j * rng.normal(size=(Dl, Dr, 2, 2))

    # Left unfold/refold roundtrip
    mat_left = unfold_mpo(M, left=True)
    R_left = refold_mpo(mat_left, 2, 2, left=True)
    assert R_left.shape == M.shape
    assert np.allclose(R_left, M)

    # Right unfold/refold roundtrip
    mat_right = unfold_mpo(M, left=False)
    R_right = refold_mpo(mat_right, 2, 2, left=False)
    assert R_right.shape == M.shape
    assert np.allclose(R_right, M)


def test_pad_and_identity_embed():
    A = np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]])
    B = pad_tensor(A, (3, 4))
    assert B.shape == (3, 4)
    # identity_embed_pad with out_axes=(0,), in_axes=(1,)
    E = identity_embed_pad(A, (3, 4), (0,), (1,))
    m = A.shape[0]
    n = A.shape[1]
    unfolded = unfold_tensor(E, (0,), (1,))
    # diagonal entries beyond original min-dim must be 1
    diag_idx = np.arange(min(m, n), min(unfolded.shape))
    if diag_idx.size > 0:
        assert np.allclose(np.diag(unfolded)[min(m, n) :], 1.0)


def test_identity_pad_to_power_behaviour(rng):
    T = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    P = identity_pad_to_power(T, base=2, out_axes=(0,), in_axes=(1,))
    # dims should be powers of two
    assert P.shape[0] in (4, 8)
    assert P.shape[1] in (4, 8)
    # top-left block equals original (after unfolding)
    unfolded = unfold_tensor(P, (0,), (1,))
    orig_unfold = unfold_tensor(T, (0,), (1,))
    assert np.allclose(
        unfolded[: orig_unfold.shape[0], : orig_unfold.shape[1]], orig_unfold
    )


def test_unitary_diag_dilation_edge_cases():
    s_empty, c_empty, a_empty = unitary_diag_dilation([])
    assert isinstance(s_empty, np.ndarray) and s_empty.size == 0
    assert a_empty == 1.0

    s = np.array([0.5, 0.25])
    s_pad, c_pad, alpha = unitary_diag_dilation(s)
    assert np.isclose(alpha, 0.5)
    assert np.allclose(s_pad**2 + c_pad**2, 1.0)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4), (5, 2)])
def test_unitary_dilation_square_method(rng, shape):
    m, n = shape
    M = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))

    U, a = unitary_dilation(M, method="square")
    assert_unitary(U, atol=1e-9)
    tl = U[:m, :n]
    assert np.allclose(tl, M / a, atol=1e-9)
    svals = svd(M, compute_uv=False) if M.size > 0 else np.array([0.0])
    smax = float(svals[0]) if svals.size > 0 else 0.0
    assert a + 1e-12 >= smax


def test_unitary_dilation_branches_and_zero(rng):
    M = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    Qs, a_s = unitary_dilation(M, method="square")
    assert Qs.shape == (2 * max(3, 2), 2 * max(3, 2))
    assert np.allclose(Qs[:3, :2], M / a_s, atol=1e-9)

    Qr, a_r = unitary_dilation(M, method="rect")
    assert Qr.shape == (3 + 2, 3 + 2)
    assert np.allclose(Qr[:3, :2], M / a_r, atol=1e-9)

    with pytest.raises(ValueError):
        unitary_dilation(M, method="unknown")

    tiny = np.zeros_like(M)
    Qz, az = unitary_dilation(tiny, method="square")
    assert az == 0.0
    assert_unitary(Qz)


def test_unitary_svd_properties(rng):
    M = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    m, n = M.shape
    Q, alpha = unitary_dilation(M, method="square")
    assert_unitary(Q)
    assert np.allclose(Q[:m, :n], M / alpha, atol=1e-9)
    smax = float(np.linalg.svd(M, compute_uv=False)[0]) if M.size else 0.0
    assert alpha + 1e-12 >= smax


def test_pad_tensor_square_and_errors():
    A = np.array([[1.0 + 0j]])
    B = pad_tensor(A, (4, 4))
    assert B.shape == (4, 4)
    assert np.isclose(B[0, 0], 1.0)
    # pad_tensor requires same ndim
    with pytest.raises(ValueError):
        pad_tensor(A, (4, 4, 4))


def test_parse_axes_negative_and_none_in_axes():
    out, inn = parse_axes(
        3, (-1,), None
    )  # -1 normalizes to 2, in_axes taken as complement
    assert out == (2,)
    assert set(inn) == {0, 1}


def test_identity_embed_pad_empty_axes():
    A = np.array([[1.0 + 0j]])
    with pytest.raises(ValueError):
        identity_embed_pad(A, (2, 2), (), ())


def test_unitary_diag_dilation_invalid_alpha():
    with pytest.raises(ValueError):
        unitary_diag_dilation([np.inf, 0.5])


def test_tensor_network_encoding_simple_sequence():
    # single bond axis per site (shape: bond, phys_out, phys_in)
    A = np.ones((1, 2, 2), dtype=complex)
    V = [A, A]
    E = -np.ones((2, 2), dtype=int)
    E[0, 1], E[1, 0] = 0, 0  # connect bond axis 0 between the two sites
    T = TensorNetworkEncoding(V, E, pad_pow=2)
    ops, size, alphas = T.build_operator_sequence(unfold=True)
    assert isinstance(ops, list)
