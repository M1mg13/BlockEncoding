import pytest

from BlockEncoding import AtomicRegistry, base_count


def test_atom_size_validation_and_norm_key():
    with pytest.raises(ValueError):
        AtomicRegistry(atom_size=1)

    regs = AtomicRegistry()
    # float keys are normalized to ints via _norm_key through public API
    regs.set((1.9, 0.1), 2)
    assert len(regs.get((1, 0))) == base_count(2, regs.atom_size)


def test_grow_registry_and_assign_many_keys_triggers_grow():
    regs = AtomicRegistry()
    # allocate many single-atom registers to force registry growth
    n = 12
    for i in range(n):
        regs.set((i, 0), 2)  # base_count(2) == 1 per key
    assert regs.size >= n
    # all keys present and retrievable
    for i in range(n):
        assert isinstance(regs.get((i, 0)), tuple)


def test_set_zero_size_creates_empty_allocation_and_missing_set_behaviour():
    regs = AtomicRegistry()
    # base_count(0) == 0 -> should create an empty tuple allocation
    regs.set((5, 0), 0)
    g = regs.get((5, 0))
    assert isinstance(g, tuple) and len(g) == 0

    # ensure missing-key free raises KeyError as implemented
    with pytest.raises(KeyError):
        regs.free((99, 99))


def test_free_marks_none_and_get_after_free_raises():
    regs = AtomicRegistry()
    regs.set((0, 0), 2)
    # free existing key:
    regs.free((0, 0))
    # public get should now raise
    with pytest.raises(KeyError):
        regs.get((0, 0))


def test_free_missing_key_raises_keyerror_as_implemented():
    regs = AtomicRegistry()
    with pytest.raises(KeyError):
        regs.free((999, 999))


def test_reuse_freed_atoms_on_subsequent_set_does_not_grow_size():
    regs = AtomicRegistry()
    regs.set((10, 0), 2)
    before = regs.size
    regs.free((10, 0))
    # allocate a new key of same size, should reuse freed atom and not grow registry
    regs.set((11, 0), 2)
    assert regs.size == before


def test_resize_release_emulated_by_free_and_set_shrinks_allocation():
    regs = AtomicRegistry()
    # allocate a larger allocation then shrink it by free+set
    regs.set((20, 0), 16)  # base_count(16) == 4
    assert len(regs.get((20, 0))) == base_count(16, regs.atom_size)
    # emulate shrink: free and set smaller size
    regs.free((20, 0))
    regs.set((20, 0), 2)
    assert len(regs.get((20, 0))) == base_count(2, regs.atom_size)


def test_redistribute_missing_src_raises_keyerror_and_size_mismatch_raises_valueerror():
    regs = AtomicRegistry()
    regs.set((0, 0), 2)
    regs.set((1, 0), 1)
    # missing src key triggers KeyError in the internal get path
    with pytest.raises(KeyError):
        regs.redistribute([(0, 0), (99, 99)], 2, [(0, 1), (1, 2)])

    # mismatch in total units -> ValueError
    with pytest.raises(ValueError):
        regs.redistribute([(0, 0), (1, 0)], 0, ((0, 2), (1, 2)))
