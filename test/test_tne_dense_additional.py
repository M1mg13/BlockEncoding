import numpy as np
import pytest
from conftest import find_best_perm_and_scale

from BlockEncoding import TensorNetworkEncoding


@pytest.mark.parametrize("n_nodes", [2, 3, 4])
@pytest.mark.parametrize("trials", [5])
def test_random_small_graphs_dense_matches_brute(n_nodes, trials, rng):
    p = 2  # physical dim per site
    for _ in range(trials):
        # Random symmetric adjacency (no self-loops).
        A = np.zeros((n_nodes, n_nodes), dtype=bool)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.6:
                    A[i, j] = A[j, i] = True

        # Assign bond dims for each unordered edge
        bond_dims = {}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if A[i, j]:
                    bond_dims[(i, j)] = int(rng.integers(1, 4))

        # Neighbor ordering deterministic
        neighs = [
            sorted(j for j in range(n_nodes) if j != i and A[i, j])
            for i in range(n_nodes)
        ]

        # Build node tensors V: axes = [bond dims in neighbor order] + [p,p]
        V = []
        for i in range(n_nodes):
            axes = [bond_dims[(min(i, j), max(i, j))] for j in neighs[i]]
            axes = axes + [p, p]
            T = (
                rng.normal(size=tuple(axes)) + 1j * rng.normal(size=tuple(axes))
            ).astype(complex)
            V.append(T)

        # Build E matrix mapping neighbor positions to axis indices
        E = -np.ones((n_nodes, n_nodes), dtype=int)
        for i in range(n_nodes):
            for ai, j in enumerate(neighs[i]):
                E[i, j] = ai

        tne = TensorNetworkEncoding(V, E, pad_pow=2)
        G, gamma = tne.build_dense_matrix(max_dims=1 << 18)
        assert np.allclose(G.conj().T @ G, np.eye(G.shape[0]), atol=1e-9)

        H_net = TensorNetworkEncoding(V, E).brute_contract()
        phys_total = 2**n_nodes
        G_block = (gamma * G[:phys_total, :phys_total]).astype(complex)

        res, perm, s = find_best_perm_and_scale(H_net, G_block)
        assert (
            res <= 1e-7
        ), f"Random graph failed residual={res}, perm={perm}, scale={s}"


def test_disconnected_components_dense_product_structure(rng):
    p = 2
    # component A: single site with random 2x2
    A0 = (rng.normal(size=(p, p)) + 1j * rng.normal(size=(p, p))).astype(complex)
    # component B: two-site chain
    core0 = (rng.normal(size=(1, 1, p, p)) + 1j * rng.normal(size=(1, 1, p, p))).astype(
        complex
    )
    core1 = (rng.normal(size=(1, 1, p, p)) + 1j * rng.normal(size=(1, 1, p, p))).astype(
        complex
    )

    V = [A0, core0[0], core1[0]]
    E = -np.ones((3, 3), dtype=int)
    E[1, 2] = 0
    E[2, 1] = 0

    tne = TensorNetworkEncoding(V, E, pad_pow=2)
    G, gamma = tne.build_dense_matrix(max_dims=1 << 14)
    assert np.allclose(G.conj().T @ G, np.eye(G.shape[0]), atol=1e-9)

    H_net = TensorNetworkEncoding(V, E).brute_contract()
    phys_total = 2**3
    G_block = (gamma * G[:phys_total, :phys_total]).astype(complex)

    res, perm, s = find_best_perm_and_scale(H_net, G_block)
    assert (
        res <= 1e-8
    ), f"Disconnected components mismatch residual={res}, perm={perm}, scale={s}"


def test_constructor_rejects_pad_pow_none():
    p = 2
    V = [np.ones((p, p), dtype=complex)]
    E = -np.ones((1, 1), dtype=int)
    with pytest.raises(Exception):
        TensorNetworkEncoding(V, E, pad_pow=None)


@pytest.mark.parametrize("bond_dim", [3, 4])
def test_large_bond_small_network_matches_brute(rng, bond_dim):
    n = 3
    p = 2
    cores = []
    for i in range(n):
        Dl = 1 if i == 0 else bond_dim
        Dr = 1 if i == n - 1 else bond_dim
        core = (
            rng.normal(size=(Dl, Dr, p, p)) + 1j * rng.normal(size=(Dl, Dr, p, p))
        ).astype(complex)
        cores.append(core)

    tne = TensorNetworkEncoding.mpo_to_tensor_network(
        cores,
        lc=np.ones(cores[0].shape[0], dtype=complex),
        rc=np.ones(cores[-1].shape[1], dtype=complex),
        pad_pow=2,
    )
    G, gamma = tne.build_dense_matrix(max_dims=1 << 20)
    assert np.allclose(G.conj().T @ G, np.eye(G.shape[0]), atol=1e-9)

    H_net = tne.brute_contract()
    phys_total = 2**n
    G_block = (gamma * G[:phys_total, :phys_total]).astype(complex)

    res, perm, s = find_best_perm_and_scale(H_net, G_block)
    assert (
        res <= 1e-7
    ), f"Large bond small network mismatch residual={res}, perm={perm}, scale={s}"
