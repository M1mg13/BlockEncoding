import numpy as np
import pytest
from conftest import assert_unitary, brute_contract_mpo

from BlockEncoding import TensorNetworkEncoding


@pytest.mark.parametrize("bond_dim", [1, 2])
def test_build_dense_three_qubits_chain(rng, bond_dim):
    # 3 qubits (physical dim 2 each), small bond dims to exercise registry paths
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
    lc = np.ones(cores[0].shape[0], dtype=complex)
    rc = np.ones(cores[-1].shape[1], dtype=complex)

    tne = TensorNetworkEncoding.mpo_to_tensor_network(cores, lc=lc, rc=rc, pad_pow=2)

    G, gamma = tne.build_dense_matrix(max_dims=1 << 14)
    assert_unitary(G, atol=1e-9)

    H = brute_contract_mpo(cores, lc, rc)
    phys_total = 2**n
    assert np.allclose(gamma * G[:phys_total, :phys_total], H, atol=1e-8)


@pytest.mark.parametrize("bond_dim", [1, 2])
def test_build_dense_five_qubits_chain(rng, bond_dim):
    # 5 qubits, bond_dim=1,2 (still nontrivial exponential physical block)
    n = 5
    p = 2
    cores = []
    for i in range(n):
        Dl = 1 if i == 0 else bond_dim
        Dr = 1 if i == n - 1 else bond_dim
        core = (
            rng.normal(size=(Dl, Dr, p, p)) + 1j * rng.normal(size=(Dl, Dr, p, p))
        ).astype(complex)
        cores.append(core)
    lc = np.ones(cores[0].shape[0], dtype=complex)
    rc = np.ones(cores[-1].shape[1], dtype=complex)

    tne = TensorNetworkEncoding.mpo_to_tensor_network(cores, lc=lc, rc=rc, pad_pow=2)

    G, gamma = tne.build_dense_matrix(max_dims=1 << 16)
    assert_unitary(G, atol=1e-9)

    H = brute_contract_mpo(cores, lc, rc)
    phys_total = 2**n

    assert np.allclose(gamma * G[:phys_total, :phys_total], H, atol=1e-8)


@pytest.mark.parametrize("opt", [(3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2)])
def test_build_dense_clique(rng, opt):
    """
    Build a full clique on n nodes where each edge has equal bond_dim.
    Compare TensorNetworkEncoding.build_dense_matrix top-left block to explicit contraction.
    """
    n, bond_dim = opt
    p = 2  # physical dim per site
    # Build node tensors: each node has (n-1) bond axes then phys_out, phys_in
    V = []
    for i in range(n):
        axes = [bond_dim] * (n - 1) + [p, p]
        T = rng.normal(size=tuple(axes)) + 1j * rng.normal(size=tuple(axes))
        V.append(T.astype(complex))

    # Build E: for node i, neighbor ordering is [0..i-1, i+1..n-1] (consistent)
    E = -np.ones((n, n), dtype=int)
    for i in range(n):
        neighs = [j for j in range(n) if j != i]
        for ai, j in enumerate(neighs):
            E[i, j] = ai
    # Validate symmetric mapping: ensure E[j,i] points to matching axis index
    # For our construction, node j's neighbor list index of i:
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            idx = list(k for k in range(n) if k != j)
            E[j, i] = idx.index(i)

    tne = TensorNetworkEncoding(V, E, pad_pow=2)
    G, gamma = tne.build_dense_matrix(max_dims=1 << 20)
    assert_unitary(G, atol=1e-9)

    # verify E symmetry and axis ranges
    for i in range(n):
        for j in range(n):
            if E[i, j] >= 0:
                assert E[j, i] >= 0, f"half-edge {i},{j}"
                di = V[i].shape[E[i, j]]
                dj = V[j].shape[E[j, i]]
                assert (
                    di == dj
                ), f"bond dim mismatch {i}.{E[i,j]}={di} vs {j}.{E[j,i]}={dj}"

    H = TensorNetworkEncoding(V, E).brute_contract()
    phys_total = 2**n
    assert np.allclose(gamma * G[:phys_total, :phys_total], H, atol=1e-8)


def test_build_dense_mixed_couplings_triangle(rng):
    """
    3-node triangle where each edge gets a different bond dimension (1,2,3).
    This tests non-uniform bond sizes and ordering correctness.
    """
    p = 2
    # set edge dims for edges (0-1, 1-2, 2-0)
    d01, d12, d20 = 1, 2, 3
    # Node 0 connects to 1 (axis 0) and 2 (axis 1) -> axes sizes [d01, d20]
    V = []
    V.append(
        (
            rng.normal(size=(d01, d20, p, p)) + 1j * rng.normal(size=(d01, d20, p, p))
        ).astype(complex)
    )
    # Node1 connects to 0 (axis 0) and 2 (axis 1)
    V.append(
        (
            rng.normal(size=(d01, d12, p, p)) + 1j * rng.normal(size=(d01, d12, p, p))
        ).astype(complex)
    )
    # Node2 connects to 0 (axis 0) and 1 (axis 1)
    V.append(
        (
            rng.normal(size=(d20, d12, p, p)) + 1j * rng.normal(size=(d20, d12, p, p))
        ).astype(complex)
    )

    E = -np.ones((3, 3), dtype=int)
    # Node ordering of bond axes chosen consistently above
    E[0, 1] = 0
    E[1, 0] = 0
    E[1, 2] = 1
    E[2, 1] = 1
    E[2, 0] = 0
    E[0, 2] = 1

    tne = TensorNetworkEncoding(V, E, pad_pow=2)
    G, gamma = tne.build_dense_matrix(max_dims=1 << 16)
    assert_unitary(G, atol=1e-9)

    H = TensorNetworkEncoding(V, E).brute_contract()
    phys_total = 2**3
    assert np.allclose(gamma * G[:phys_total, :phys_total], H, atol=1e-8)
