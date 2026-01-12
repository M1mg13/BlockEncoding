import numpy as np
import pytest
from conftest import (
    assert_unitary,
    brute_contract_mpo,
    find_best_perm_and_scale,
    qubo_direct_dense,
)

from BlockEncoding import (
    QUBONetworkEncoding,
    QUBOPauliSumEncoding,
    QUBOSweepEncoding,
    TensorNetworkEncoding,
)


@pytest.mark.parametrize("bond_dim", [1, 2])
def test_mpo_chain_dense_vs_mpo_brute(rng, bond_dim):
    n = 3
    p = 2
    cores = []
    for i in range(n):
        Dl = 1 if i == 0 else bond_dim
        Dr = 1 if i == n - 1 else bond_dim
        core = rng.normal(size=(Dl, Dr, p, p)) + 1j * rng.normal(size=(Dl, Dr, p, p))
        cores.append(core.astype(complex))
    lc = np.ones(cores[0].shape[0], dtype=complex)
    rc = np.ones(cores[-1].shape[1], dtype=complex)

    tne = TensorNetworkEncoding.mpo_to_tensor_network(cores, lc=lc, rc=rc, pad_pow=2)
    G, gamma = tne.build_dense_matrix(max_dims=1 << 14)
    assert_unitary(G, atol=1e-9)

    H_mpo = brute_contract_mpo(cores, lc, rc)
    H_net = tne.brute_contract()

    phys_total = 2**n
    G_block = (gamma * G[:phys_total, :phys_total]).astype(complex)

    res, perm, s = find_best_perm_and_scale(G_block, H_mpo)
    assert res <= 1e-7, (
        f"MPO chain mismatch: residual={res}, best_perm={perm}, best_scale={s}\n"
        f"||G||={np.linalg.norm(G_block)}, ||H_mpo||={np.linalg.norm(H_mpo)}"
    )

    assert np.allclose(H_mpo, H_net, atol=1e-12)


@pytest.mark.parametrize("bond_dims", [(1, 2, 3)])
def test_triangle_mixed_dims_matches_brute(rng, bond_dims):
    d01, d12, d20 = bond_dims
    p = 2
    V0 = (
        rng.normal(size=(d01, d20, p, p)) + 1j * rng.normal(size=(d01, d20, p, p))
    ).astype(complex)
    V1 = (
        rng.normal(size=(d01, d12, p, p)) + 1j * rng.normal(size=(d01, d12, p, p))
    ).astype(complex)
    V2 = (
        rng.normal(size=(d20, d12, p, p)) + 1j * rng.normal(size=(d20, d12, p, p))
    ).astype(complex)
    V = [V0, V1, V2]
    E = -np.ones((3, 3), dtype=int)
    E[0, 1], E[1, 0] = 0, 0
    E[1, 2], E[2, 1] = 1, 1
    E[2, 0], E[0, 2] = 0, 1

    tne = TensorNetworkEncoding(V, E, pad_pow=2)
    G, gamma = tne.build_dense_matrix(max_dims=1 << 12)
    assert_unitary(G, atol=1e-9)
    G_block = (gamma * G[: 2**3, : 2**3]).astype(complex)

    H_net = TensorNetworkEncoding(V, E).brute_contract()
    res, perm, s = find_best_perm_and_scale(G_block, H_net)
    assert res <= 1e-7, f"Triangle mismatch residual={res}, perm={perm}, scale={s}"


@pytest.mark.parametrize("n,bond_dim", [(3, 1), (3, 2), (4, 1)])
def test_clique_small_matches_brute(rng, n, bond_dim):
    p = 2
    V = []
    for i in range(n):
        axes = [bond_dim] * (n - 1) + [p, p]
        T = rng.normal(size=tuple(axes)) + 1j * rng.normal(size=tuple(axes))
        V.append(T.astype(complex))
    E = -np.ones((n, n), dtype=int)
    for i in range(n):
        neighs = [j for j in range(n) if j != i]
        for ai, j in enumerate(neighs):
            E[i, j] = ai
    tne = TensorNetworkEncoding(V, E, pad_pow=2)
    G, gamma = tne.build_dense_matrix(max_dims=1 << 20)
    assert_unitary(G, atol=1e-9)
    G_block = (gamma * G[: 2**n, : 2**n]).astype(complex)
    H_net = TensorNetworkEncoding(V, E).brute_contract()
    res, perm, s = find_best_perm_and_scale(G_block, H_net)
    assert res <= 1e-7, f"Clique mismatch residual={res}, perm={perm}, scale={s}"


@pytest.mark.parametrize(
    "encoder",
    [QUBOSweepEncoding, QUBOPauliSumEncoding, QUBONetworkEncoding],
)
@pytest.mark.parametrize("n", range(1, 5))
def test_qubo_encoders_produce_same_dense(encoder, n, rng):
    Q = rng.normal(size=(n, n))
    c = 0.123

    H_direct = qubo_direct_dense(Q, c=c)
    tne = encoder(Q, c=c, tol=1e-12)

    G, gamma = tne.build_dense_matrix(max_dims=1 << 16, tol=1e-12)
    assert_unitary(G, atol=1e-9)
    phys_total = H_direct.shape[0]
    G_block = (gamma * G[:phys_total, :phys_total]).astype(complex)

    res, perm, s = find_best_perm_and_scale(H_direct, G_block)
    assert res <= 1e-8, (
        f"QUBO encoder {encoder.__name__} mismatch n={n}: residual={res}, perm={perm}, scale={s}\n"
        f"||H_direct||={np.linalg.norm(H_direct)}, ||G_block||={np.linalg.norm(G_block)}"
    )
