import numpy as np
import pytest
from conftest import assert_unitary

from BlockEncoding import SiteLayer


def test_unitary_shortcircuit_and_force_expand():
    T = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float) / np.sqrt(2.0)
    layer = SiteLayer(T, pad_pow=2)

    # default: should short-circuit to diagonal normalized core (s == 1)
    op = layer.factor(out_axes=(0,), in_axes=(1,))
    assert pytest.approx(op.beta, rel=1e-12) == 1.0
    # C should be diagonal of ones (k == lcm(m, n) == 2)
    assert op.C.shape == (2, 2)
    assert np.allclose(np.diag(op.C), 1.0)
    assert_unitary(op.U)
    assert_unitary(op.Vh)

    # force expansion: always_expand_core -> should produce a block-core (Su)
    op2 = layer.factor(out_axes=(0,), in_axes=(1,), always_expand_core=True)
    # now C must be a larger block (2*k x 2*k for m=n=2 => 4x4)
    assert op2.C.shape == (4, 4)
    # must remain unitary
    assert_unitary(op2.C)


def test_zero_beta_path_produces_unitary_core():
    T = np.zeros((2, 2), dtype=complex)
    layer = SiteLayer(T, pad_pow=2)
    op = layer.factor(out_axes=(0,), in_axes=(1,))
    # zero matrix => beta == 0 and core C should still be unitary
    assert op.beta == 0.0
    assert_unitary(op.C)
    # U and Vh should be square unitary matrices (SVD of zeros yields orthonormal factors)
    assert_unitary(op.U)
    assert_unitary(op.Vh)


def test_pad_pow_none_rectangular_and_unitary_match_core():
    # rectangular unfolded matrix m=2, n=4: use a tensor shaped (2,4)
    rng = np.random.default_rng(42)
    T = (rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))).astype(complex)
    layer = SiteLayer(T, pad_pow=None)

    # Request matching of unitaries to core and no pre-padding (pad_pow=None branch)
    op = layer.factor(out_axes=(0,), in_axes=(1,), unitary_match_core=True)

    assert op.m == 2
    assert op.n == 4
    # lcm(2,4)=4, x == 2 when pad_pow is None -> final core size should be 2*k
    k = np.lcm(op.m, op.n)
    assert op.x == 2
    assert op.C.shape == (op.x * k, op.x * k)
    # core must be unitary dilation
    assert_unitary(op.C)
    # expanded U/Vh should be square and compatible with core embedding
    assert op.U.shape[0] == op.U.shape[1]
    assert op.Vh.shape[0] == op.Vh.shape[1]
    assert_unitary(op.U)
    assert_unitary(op.Vh)


def test_pad_pow_embedding_identity_embed_for_large_pad_pow():
    # m=n=3, pad_pow=4 triggers identity_pad_to_power to 4x4 pre-pad, then x==pad_pow==4 embedding
    rng = np.random.default_rng(7)
    T = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))).astype(complex)
    layer = SiteLayer(T, pad_pow=4)

    op = layer.factor(out_axes=(0,), in_axes=(1,))
    # After identity pre-pad the unfolded dims will be powers-of-4 -> m==n==4, k==4
    k = np.lcm(op.m, op.n)
    assert op.x == 4
    # Su initial size is 2*k (8), then identity_embed_pad to (x*k) == 16
    assert op.C.shape == (op.x * k, op.x * k)
    # core must be unitary
    assert_unitary(op.C)


def test_invalid_pad_method_raises():
    T = np.eye(2, dtype=complex)
    layer = SiteLayer(T, pad_pow=2)
    with pytest.raises(ValueError):
        layer.factor(out_axes=(0,), in_axes=(1,), pad_method="no-such-method")


def test_ops_unfold_false_hadamard():
    T = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float) / np.sqrt(2.0)
    layer = SiteLayer(T, pad_pow=2)

    sf = layer.factor(out_axes=(0,), in_axes=(1,))
    U, C, Vh = sf.ops(unfold=True)
    U_t, C_t, Vh_t = sf.ops(unfold=False)

    # folded views reshape back to unfolded
    assert np.allclose(U, U_t.reshape((-1, sf.m)))
    assert np.allclose(Vh, Vh_t.reshape((sf.n, -1)))
    assert np.allclose(C, C_t.reshape((sf.x * sf.d * sf.m, sf.x * sf.p * sf.n)))

    # basic unitarity checks
    assert_unitary(C)
    assert_unitary(U)
    assert_unitary(Vh)


def test_ops_unfold_false_rect_pad_none(rng):
    # rectangular example: m=2, n=4 -> pad_pow=None branch
    T = (rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))).astype(complex)
    layer = SiteLayer(T, pad_pow=None)

    sf = layer.factor(out_axes=(0,), in_axes=(1,), unitary_match_core=False)
    U, C, Vh = sf.ops(unfold=True)
    U_t, C_t, Vh_t = sf.ops(unfold=False)

    # folded views reshape back to unfolded
    assert np.allclose(U, U_t.reshape((-1, sf.m)))
    assert np.allclose(Vh, Vh_t.reshape((sf.n, -1)))
    assert np.allclose(C, C_t.reshape((sf.x * sf.d * sf.m, sf.x * sf.p * sf.n)))

    # shapes consistent with metadata
    assert U_t.shape == (*sf.out_shape, sf.m)
    assert Vh_t.shape == (sf.n, *sf.in_shape)
    assert C_t.shape == (sf.x, sf.d, sf.m, sf.x, sf.p, sf.n)

    # dilation/core should be unitary
    assert_unitary(C)
