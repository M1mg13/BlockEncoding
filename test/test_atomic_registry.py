import pytest

from BlockEncoding import AtomicRegistry, base_count


def test_init_invalid_atom_size_raises():
    with pytest.raises(ValueError):
        AtomicRegistry(atom_size=1)


def test_set_assigns_and_grows_registry():
    regs = AtomicRegistry()
    assert regs.size == 0
    # request a register that needs multiple atoms (2^4 = 16 -> base_count=4)
    assigned = regs.set((0, 0), 16)
    assert regs.size >= 4
    # stored allocation length equals base_count
    assert len(regs.get((0, 0))) == base_count(16, regs.atom_size)
    # returned value is a tuple of atom indices
    assert isinstance(assigned, tuple)


def test_set_get_free_behaviour_without_resize():
    regs = AtomicRegistry()
    regs.set((0, 0), 3)
    regs.set((1, 0), 2)

    assert len(regs.get((0, 0))) == base_count(3, regs.atom_size)
    assert len(regs.get((1, 0))) == base_count(2, regs.atom_size)

    # emulate resize down: free and re-set with smaller size
    regs.free((0, 0))
    regs.set((0, 0), 1)
    assert len(regs.get((0, 0))) == base_count(1, regs.atom_size)

    # emulate resize up: free and re-set with larger size
    regs.free((1, 0))
    regs.set((1, 0), 4)
    assert len(regs.get((1, 0))) == base_count(4, regs.atom_size)

    # free releases and marks key as deleted; subsequent get should raise
    regs.free((1, 0))
    with pytest.raises(KeyError):
        regs.get((1, 0))


def test_set_zero_size_creates_empty_allocation():
    regs = AtomicRegistry()
    # base_count(0) == 0 -> should create an empty tuple allocation
    regs.set((5, 0), 0)
    g = regs.get((5, 0))
    assert isinstance(g, tuple) and len(g) == 0


def test_norm_key_accepts_and_normalizes_numeric_tuple():
    regs = AtomicRegistry()
    regs.set((1.9, 0.1), 2)  # should normalize to (1,0)
    assert len(regs.get((1, 0))) == base_count(2, regs.atom_size)


def test_redistribute_success_and_original_keys_deleted():
    regs = AtomicRegistry()
    regs.set((0, 0), 2)
    regs.set((1, 0), 1)
    # redistribute into new site key k=2 with dst entries sizes that match
    regs.redistribute([(0, 0), (1, 0)], 2, [(0, 1), (1, 2)])
    # (k=2, j=0) got base_count(1), (k=2, j=1) got base_count(2)
    assert len(regs.get((2, 0))) == base_count(1, regs.atom_size)
    assert len(regs.get((2, 1))) == base_count(2, regs.atom_size)
    # original keys are removed (access should raise)
    with pytest.raises(KeyError):
        regs.get((0, 0))
    with pytest.raises(KeyError):
        regs.get((1, 0))


def test_redistribute_mismatch_raises():
    regs = AtomicRegistry()
    regs.set((0, 0), 2)
    regs.set((1, 0), 1)
    # request redistribution where dst total atoms != src total atoms
    with pytest.raises(ValueError):
        regs.redistribute([(0, 0), (1, 0)], 0, ((0, 2), (1, 2)))


def test_free_missing_key_raises_keyerror():
    regs = AtomicRegistry()
    # free on a key that was never set triggers a KeyError in current implementation
    with pytest.raises(KeyError):
        regs.free((99, 99))
