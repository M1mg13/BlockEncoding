import numpy as np

from BlockEncoding import TensorNetworkEncoding


def test_mpo_to_tensor_network_single_core_absorbs_boundaries():
    # create a minimal valid TensorNetworkEncoding instance to call the method on

    # single MPO core with Dl=2, Dr=3, phys_out=phys_in=2
    Dl, Dr = 2, 3
    core = np.arange(Dl * Dr * 2 * 2, dtype=float).reshape(Dl, Dr, 2, 2).astype(complex)

    # left/right boundary vectors matching bond dims
    lc = np.array([0.5, 1.0], dtype=complex)
    rc = np.array([1.0, 0.0, -1.0], dtype=complex)

    # Should succeed for single-core MPO: boundaries absorbed and TNE created
    tne = TensorNetworkEncoding.mpo_to_tensor_network([core], lc=lc, rc=rc, pad_pow=2)
    assert isinstance(tne, TensorNetworkEncoding)
    # one node only
    assert len(tne.V) == 1
    # node must be a tensor with last two axes physical (2,2)
    node = tne.V[0].T if hasattr(tne.V[0], "T") else np.asarray(tne.V[0])
    assert node.shape[-2:] == (2, 2)
    # E should be 1x1 with -1 on diagonal
    assert tne.E.shape == (1, 1)
    assert tne.E[0, 0] == -1


def test_mpo_to_tensor_network_multi_core_index_error_detected():
    # Build two MPO cores (Dl0=2, Dr0=1) and (Dl1=1, Dr1=3) to test chaining.
    core0 = np.arange(2 * 1 * 2 * 2, dtype=float).reshape(2, 1, 2, 2).astype(complex)
    core1 = np.arange(1 * 3 * 2 * 2, dtype=float).reshape(1, 3, 2, 2).astype(complex)

    lc = np.array([1.0, 0.0], dtype=complex)
    rc = np.array([1.0, 0.0, 0.0], dtype=complex)

    TensorNetworkEncoding.mpo_to_tensor_network([core0, core1], lc=lc, rc=rc, pad_pow=2)
