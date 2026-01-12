import numpy as np
import pytest
from conftest import assert_unitary, qubo_direct_dense

from BlockEncoding import (
    QUBONetworkEncoding,
    QUBOPauliSumEncoding,
    QUBOSweepEncoding,
)


@pytest.mark.parametrize(
    "encoder",
    [
        QUBOSweepEncoding,
        QUBOPauliSumEncoding,
        QUBONetworkEncoding,
    ],
)
@pytest.mark.parametrize("n", range(1, 5))
def test_qubo_encoders_same_dense(rng, encoder, n):
    Q = rng.normal(size=(n, n))
    c = 0.123

    # direct dense operator (uses shared helper with LSB bit ordering)
    H_direct = qubo_direct_dense(Q, c=c)
    phys_total = H_direct.shape[0]

    # run encoder and convert to TensorNetworkEncoding
    TNE = encoder(Q, c=c, tol=1e-12)

    # build global unitary and top-left physical block
    G, gamma = TNE.build_dense_matrix(max_dims=1 << 16, tol=1e-12)
    assert_unitary(G, atol=1e-9)

    # compare top-left physical block
    assert np.allclose(gamma * G[:phys_total, :phys_total], H_direct, atol=1e-9)


def test_qubo_encoders_sparse_n4(rng):
    n = 4
    Q = np.zeros((n, n))
    # sparse random off-diagonals
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                v = float(rng.normal())
                Q[i, j] = Q[j, i] = v
    c = 0.7

    for Enc in (QUBOSweepEncoding, QUBOPauliSumEncoding):
        tne = Enc(Q, c=c, tol=1e-12)
        G, gamma = tne.build_dense_matrix(max_dims=1 << 18)
        H = qubo_direct_dense(Q, c=c)
        phys = H.shape[0]
        assert np.allclose(gamma * G[:phys, :phys], H, atol=1e-8)
