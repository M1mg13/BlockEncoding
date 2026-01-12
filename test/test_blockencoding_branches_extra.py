import numpy as np
import pytest

from BlockEncoding import (
    TensorNetworkEncoding,
    identity_embed_pad,
    pad_tensor,
    parse_axes,
    refold_mpo,
    refold_tensor,
    unfold_mpo,
    unfold_tensor,
    unitary_diag_dilation,
    unitary_dilation,
)


def test_parse_axes_bad_inputs_raise_or_normalize():
    # duplicate out_axes -> error
    with pytest.raises(ValueError):
        parse_axes(3, (0, 0), None)

    # out_axes out of range -> error
    with pytest.raises(ValueError):
        parse_axes(2, (3,), None)

    # negative index normalization: should normalize rather than raise
    out, inn = parse_axes(3, (-1,), None)
    assert out == (2,)
    assert set(inn) == {0, 1}


def test_unfold_refold_tensor_shape_mismatch_and_roundtrip(rng):
    T = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    out_axes = (0, 2)
    in_axes = (1,)
    M = unfold_tensor(T, out_axes, in_axes)
    # correct refold is okay
    Trec = refold_tensor(M, T.shape, out_axes, in_axes)
    assert np.allclose(Trec, T)
    # wrong shape (mismatched full_shape) should raise
    with pytest.raises(ValueError):
        refold_tensor(M, (2, 3, 5), out_axes, in_axes)


def test_unfold_refold_mpo_roundtrip(rng):
    M = rng.normal(size=(3, 4, 2, 2)) + 1j * rng.normal(size=(3, 4, 2, 2))
    # left True roundtrip
    L = unfold_mpo(M, left=True)
    R = refold_mpo(L, 2, 2, left=True)
    assert np.allclose(R, M)
    # left False roundtrip
    L2 = unfold_mpo(M, left=False)
    R2 = refold_mpo(L2, 2, 2, left=False)
    assert np.allclose(R2, M)


def test_identity_embed_pad_invalid_and_diagonal_padding():
    A = np.eye(3, dtype=complex)
    # dims too small for A -> should raise
    with pytest.raises(ValueError):
        identity_embed_pad(A, (2, 2), (0,), (1,))
    # valid pad and check extra diagonals are ones
    P = identity_embed_pad(A, (4, 4), (0,), (1,))
    U = unfold_tensor(P, (0,), (1,))
    diag_tail = np.diag(U)[3:]
    assert np.allclose(diag_tail, 1.0)


def test_pad_tensor_dimension_mismatch_raises():
    A = np.zeros((2, 3))
    with pytest.raises(ValueError):
        pad_tensor(A, (4,))  # wrong length dims


def test_unitary_diag_dilation_infinite_and_zero_cases():
    # infinite singular should raise ValueError
    with pytest.raises(ValueError):
        unitary_diag_dilation([np.inf, 0.5])
    # zeros produce alpha==0 branch
    s = np.array([0.0, 0.0])
    s_pad, c_pad, alpha = unitary_diag_dilation(s, tol=1e-12)
    assert alpha == 0.0
    assert np.allclose(s_pad, 0.0)
    assert np.allclose(c_pad, 1.0)


def test_unitary_dilation_methods_and_invalid_method(rng):
    M = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    Qs, a_s = unitary_dilation(M, method="square")
    assert Qs.shape[0] == Qs.shape[1]
    Qr, a_r = unitary_dilation(M, method="rect")
    assert Qr.shape[0] == Qr.shape[1]
    with pytest.raises(ValueError):
        unitary_dilation(M, method="no-such-method")
    # tiny zero matrix -> alpha zero path
    Z = np.zeros((2, 4), dtype=complex)
    Qz, az = unitary_dilation(Z, method="square")
    assert az == 0.0
    assert Qz.shape[0] == Qz.shape[1]


def test_tensor_network_encoding_validator_rejects_bad_E_and_bond_mismatch():
    # Build V with inconsistent bond dims to trigger validation error in constructor
    A = np.ones((1, 2, 2), dtype=complex)
    B = np.ones((2, 2, 2), dtype=complex)  # left bond 2 mismatches A's right bond 1
    V = [A, B]
    E = -np.ones((2, 2), dtype=int)
    E[0, 1], E[1, 0] = 0, 0
    with pytest.raises(ValueError):
        TensorNetworkEncoding(V, E, pad_pow=2)
