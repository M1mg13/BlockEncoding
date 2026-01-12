#!/usr/bin/env python3
# Copyright 2025 Fraunhofer
# Licensed under the Apache License, Version 2.0 (the "License");
"""
BlockEncoding.py

Block-encoding from tensor-train / tensor-network cores.

Purpose
-------
Prepare numerical ingredients to realize a block-encoding of an operator H
as the top-left block (up to an explicit global scale Gamma) of a unitary
built from local, per-site dilations. This module canonicalizes per-site
SVD-based cores, tracks per-site scale factors and ancilla needs, and emits
per-site unitary blocks and register-mapping metadata for subsequent circuit
synthesis. It prepares structured matrices/tensors and bookkeeping; it is not
a gate-synthesizer.

Key behaviors and conventions
----------------------------
- Site factoring (SiteLayer.factor) computes a full SVD of the unfolded site

  tensor and produces a SiteFactor containing (U, C, Vh, beta, dims). C is a
  dilated block-core that implements a unitary dilation of M / beta, where
  beta = max singular value for that site. Callers must accumulate per-site
  betas into a global gamma and apply that global scale when realizing a
  full unitary/block-encoding.


- Padding policy:
  * pad_pow (per-site or global) can be None or an integer base (>1).

    If pad_pow is not None, per-site axes may be identity-padded to powers
    of that base before unfolding to make later qubit-mapping simpler.

  * Two distinct paddings appear in the factoring pipeline:
    - "drop" (zero) padding: when the unfolded row/column shapes differ (m < n),

      singular-value vector s_raw is extended with zeros to represent dropped
      directions; these zeros are treated as true zero singulars.

    - "identity" padding: applied after normalization (s / beta) to extend the

      normalized singular vector to the canonical length k = lcm(m, n). This
      padding uses ones so that the top-left embedding remains an identity on
      the padded directions (keeps the dilation unitary-friendly).
  Only "identity" pad_method is currently implemented; the code is structured
  to add alternate strategies later.


- Core dilation semantics:
  * unitary_diag_dilation produces normalized singulars s_norm in [0,1] and

    complementary c = sqrt(1 - s_norm^2). The dilated core C is assembled as
    Su = [[S, C], [C, -S]] (or the diagonal S for purely unitary sites).

  * factor returns C such that combining U, C, Vh with the scalar beta yields

    an operator whose top-left block equals the original unfolded matrix M.


- Ordering:

  Atomic/register ordering and LSB/MSB conventions used elsewhere in this
  module remain unchanged; see the registry and operator composition helpers
  for details.
"""
import math
import shutil
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

np.set_printoptions(
    linewidth=shutil.get_terminal_size().columns - 10,
    precision=6,
    suppress=True,
    edgeitems=10,
    threshold=1000,
)


# -------------------------
# Utilities
# -------------------------


def base_count(x: int, base: int = 2) -> int:
    """
    Returns the needed number of digits in base.
    Special short path for base == 2.
    """
    x = int(x)
    if x <= 1:
        return 0
    if base == 2:
        return int((x - 1).bit_length())
    return int(math.ceil(math.log(x, base)))


def next_pow(x: int, base: int = 2) -> int:
    """Returns next power of base not smaller than x."""
    return base ** base_count(x, base)


# --- Axis parsing and tensor (un)folding helpers ----------------------------


def _normalize_axes(axes: Sequence[int], ndim: int) -> Tuple[int, ...]:
    return tuple(a if a >= 0 else ndim + a for a in axes)


def parse_axes(
    ndim: int, out_axes: Sequence[int], in_axes: Sequence[int] = None
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Normalize and validate axes; return (out_axes, in_axes).
    If in_axes is None, it is taken as the complement of out_axes in increasing order.
    """
    out = _normalize_axes(tuple(out_axes), ndim)
    out_set = set(out)
    if len(out_set) < len(out):
        raise ValueError("out_axes must be unique")
    if out_set and max(out_set) >= ndim:
        raise ValueError("out_axes out of range")

    if in_axes is None:
        inn = tuple(i for i in range(ndim) if i not in out_set)
    else:
        inn = _normalize_axes(tuple(in_axes), ndim)
        inn_set = set(inn)
        if len(inn_set) < len(inn):
            raise ValueError("in_axes must be unique")
        if inn_set and max(inn_set) >= ndim:
            raise ValueError("in_axes out of range")
        if out_set & inn_set:
            raise ValueError("out_axes and in_axes must be disjoint")
        if len(out_set) + len(inn_set) < ndim:
            raise ValueError("out_axes/in_axes must partition all axes")
    return out, inn


def _unfold_tensor(
    T: np.ndarray, out_axes_parsed: Sequence[int], in_axes_parsed: Sequence[int]
) -> np.ndarray:
    """
    Fold expects parsed axes tuples (out_axes, in_axes).
    Returns matrix with rows = prod(out_dims), cols = prod(in_dims).
    """
    out_size = (
        int(np.prod([T.shape[i] for i in out_axes_parsed])) if out_axes_parsed else 1
    )
    in_size = (
        int(np.prod([T.shape[i] for i in in_axes_parsed])) if in_axes_parsed else 1
    )
    return T.transpose(out_axes_parsed + in_axes_parsed).reshape(out_size, in_size)


def unfold_tensor(
    T: np.ndarray, out_axes: Sequence[int], in_axes: Sequence[int] = None
) -> np.ndarray:
    """Public wrapper: normalize axes once, then call internal unfold."""
    T = np.asarray(T)
    out, inn = parse_axes(T.ndim, out_axes, in_axes)
    return _unfold_tensor(T, out, inn)


def unfold_mpo(M: np.ndarray, left: bool = True) -> np.ndarray:
    """
    MPO helper for core shaped (Dl, Dr, phys_out, phys_in).

    If left True: rows=(Dl, phys_out) -> out_axes=(0,2), cols=(Dr, phys_in) -> in_axes=(1,3)
    If left False: rows=(Dr, phys_out) -> out_axes=(1,2), cols=(Dl, phys_in) -> in_axes=(0,3)
    """
    return (
        _unfold_tensor(np.asarray(M), (0, 2), (1, 3))
        if left
        else _unfold_tensor(np.asarray(M), (1, 2), (0, 3))
    )


# --- Refold helpers -----------------------------------------------


def _refold_tensor(
    M: np.ndarray,
    dims: Sequence[int],
    out_axes_parsed: Sequence[int],
    in_axes_parsed: Sequence[int],
) -> np.ndarray:
    """
    Refold: M has shape (prod(out_dims), prod(in_dims)).
    dims is full target shape (len == ndim).
    out_axes_parsed + in_axes_parsed must partition range(ndim).
    """
    axes = out_axes_parsed + in_axes_parsed
    return M.reshape(tuple(dims[a] for a in axes)).transpose(np.argsort(axes))


def refold_tensor(
    M: np.ndarray,
    dims: Sequence[int],
    axes: Sequence[int],
    axes_sec: Sequence[int] = None,
):
    """
    Public wrapper. Accepts either:
      - axes as a single concatenated sequence (out_axes + in_axes), or
      - axes as a pair out_axes, in_axes via axes_sec

    dims must be the full target dims tuple (length == ndim).
    """
    out_axes_parsed, in_axes_parsed = parse_axes(len(dims), axes, axes_sec)
    return _refold_tensor(np.asarray(M), tuple(dims), out_axes_parsed, in_axes_parsed)


def refold_mpo(M: np.ndarray, a: int, b: int = None, left: bool = True):
    if b is None:
        b = a
    m, n = M.shape
    if not m % a == 0 or not n % b == 0:
        raise ValueError(
            f"mpo can not refold to the given dimensions {a}, {b} for {n}, {m}"
        )
    dims = (m // a, n // b, a, b) if left else (n // b, m // a, a, b)
    return (
        _refold_tensor(M, dims, (0, 2), (1, 3))
        if left
        else _refold_tensor(M, dims, (1, 2), (0, 3))
    )


# --- Identity-padding helpers (out/in naming) ------------------------------


def pad_tensor(M: np.ndarray, dims: Sequence[int]):
    """
    M shape (d0, ...,dr) -> pad to dims with zeros.
    """
    M = np.asarray(M, dtype=complex)
    if len(dims) != M.ndim:
        raise ValueError("len(dims) must equal M.ndim")
    for a, b in zip(M.shape, dims):
        if a > b:
            raise ValueError(f"Can't pad to reach smaller dimension: {a} > {b}")
    return np.pad(M, tuple((0, b - a) for a, b in zip(M.shape, dims)))


def identity_embed_pad(
    M: np.ndarray,
    dims: Sequence[int],
    out_axes: Sequence[int],
    in_axes: Sequence[int] = None,
):
    """
    Pad tensor M to dims (per-axis), then unfold with (out_axes,in_axes) and set ones
    on diagonal entries that were padded (beyond original min-dimension).
    """
    M = np.asarray(M)
    out, inn = parse_axes(M.ndim, out_axes, in_axes)
    if out and inn:
        diag_old = min(
            np.prod([M.shape[d] for d in out]), np.prod([M.shape[d] for d in inn])
        )
    else:
        diag_old = 1
    M = pad_tensor(M, dims)
    M = _unfold_tensor(M, out, inn)
    idx = np.arange(diag_old, min(M.shape))
    M[idx, idx] = 1.0 + 0.0j
    return _refold_tensor(M, tuple(dims), out, inn)


def identity_pad_to_power(
    M: np.ndarray,
    base: int = 2,
    out_axes: Sequence[int] = (),
    in_axes: Sequence[int] = None,
) -> np.ndarray:
    """
    Pad each axis length up to next_pow(base) and identity-embed on the unfolded matrix
    defined by out_axes/in_axes.
    """
    M = np.asarray(M, dtype=complex)
    dims = tuple(int(next_pow(d, int(base))) for d in M.shape)
    return identity_embed_pad(M, dims, out_axes, in_axes)


def zeros_pad_to_power(
    M: np.ndarray,
    base: int = 2,
) -> np.ndarray:
    """
    Pad each axis length up to next_pow(base) with zeros.
    """
    return pad_tensor(
        np.asarray(M, dtype=complex),
        tuple(int(next_pow(d, int(base))) for d in M.shape),
    )


# -------------------------
# Unitary dilation
# -------------------------


def unitary_diag_dilation(s: Sequence[float], gamma: float = None, tol=1e-12):
    """
    Normalize a singular-value vector and compute complementary entries for diagonal dilation.

    Given s (1D array of non-negative singular values) and an optional normalization gamma:
    - If gamma is None: s is interpreted as already normalized (expected in [0,1]); returns s, c, gamma=1.
    - If gamma is provided:
        * If gamma <= tol the function returns s_pad = 0 and c_pad = 1 (top-left block will be zero).
        * If gamma < max(s) raises ValueError (gamma must be >= max(s)).
        * Otherwise returns s_norm = clip(s/gamma, 0,1), c = sqrt(1 - s_norm**2), and gamma.

    Responsibilities
    - This routine performs normalization and complementary computation only. It does NOT perform
      any length / identity / power padding. Callers must perform any desired padding after normalization
      so that the top-left embedding remains exact.

    Returns
    -------
    s_norm, c, gamma : (ndarray, ndarray, float)
        s_norm : normalized singulars in [0,1]
        c      : complementary values sqrt(1 - s_norm**2)
        gamma  : the normalization factor used (returned so callers can accumulate scales)
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return s.copy(), s.copy(), 1.0
    s_max = s.max()
    # pick gamma if not provided
    if gamma is None:
        gamma = s_max
    else:
        gamma = abs(gamma)
    if gamma <= tol:
        # Found a zero factor, so we provide basically an X with the zero factor.
        return np.zeros_like(s), np.ones_like(s), 0.0
    if gamma < s_max:
        raise ValueError(f"Gamma must be large enough to normalize s {gamma} < {s_max}")
    if not np.isfinite(gamma):
        raise ValueError(f"Invalid unitary_dilation normalization: {gamma}")

    # normalized singular values (first p), clipped to [0,1]
    s = np.clip(s / gamma, 0.0, 1.0)

    # build singular diag
    c = np.sqrt(np.maximum(0.0, 1.0 - s**2))  # Should be fine since clipped

    return s, c, gamma


def unitary_dilation(M: np.ndarray, method="square", tol=1e-12):
    """
    Normalize a singular-value vector and compute complementary entries for diagonal dilation.

    Given s (1D array of non-negative singular values) and an optional normalization gamma:
    - If gamma is None: s is interpreted as already normalized (expected in [0,1]); returns s, c, gamma=1.
    - If gamma is provided:
        * If gamma <= tol the function returns s_pad = 0 and c_pad = 1 (top-left block will be zero).
        * If gamma < max(s) raises ValueError (gamma must be >= max(s)).
        * Otherwise returns s_norm = clip(s/gamma, 0,1), c = sqrt(1 - s_norm**2), and gamma.

    Responsibilities
    - This routine performs normalization and complementary computation only. It does NOT perform
      any length / identity / power padding. Callers must perform any desired padding after normalization
      so that the top-left embedding remains exact.

    Returns
    -------
    s_norm, c, gamma : (ndarray, ndarray, float)
        s_norm : normalized singulars in [0,1]
        c      : complementary values sqrt(1 - s_norm**2)
        gamma  : the normalization factor used (returned so callers can accumulate scales)
    """
    if method not in {"square", "rect"}:
        raise ValueError(f"Unknown unitary_dilation method {method}")
    M = np.asarray(M, dtype=complex)
    m, n = M.shape

    # full SVD once
    U, s, Vh = np.linalg.svd(M, full_matrices=True)
    p, k = min(m, n), max(m, n)

    _, c, alpha = unitary_diag_dilation(np.pad(s, (0, k - p)), gamma=None, tol=tol)
    A = np.zeros_like(M) if alpha == 0 else M / alpha
    C = np.diag(c)

    if method == "square":
        A = np.pad(A, ((0, k - m), (0, k - n)))
        if n > m:
            U = np.block(
                [
                    [U, np.zeros((m, k - m), dtype=complex)],
                    [np.zeros((k - m, m), dtype=complex), np.eye(k - m, dtype=complex)],
                ]
            )
        elif n < m:
            Vh = np.block(
                [
                    [Vh, np.zeros((n, k - n), dtype=complex)],
                    [np.zeros((k - n, n), dtype=complex), np.eye(k - n, dtype=complex)],
                ]
            )
        B = U @ C @ Vh
        Q = np.block([[A, B], [B, -A]])
    elif method == "rect":
        Bm = U @ C[:m, :m] @ U.T.conj()
        Bn = Vh.T.conj() @ C[:n, :n] @ Vh
        Q = np.block([[A, Bm], [Bn, -A.T.conj()]])
    else:
        raise ValueError(f"Unknown unitary_dilation method {method}")
    return Q, alpha


# -------------------------
# High-level MPOEncoding
# -------------------------


@dataclass(frozen=True)
class SiteFactor:
    """
    Immutable container for per-site SVD + dilation metadata.

    Attributes
    ----------
    U : ndarray
        Full-unitary from SVD. Shape (m_full, m_full) where m_full equals the
        unfolded row-space dimension after any pre-unfold identity padding
        (may equal m or be larger when unitary_match_core/pad_pow expansions are used).
    C : ndarray
        Dilated core implementing a unitary dilation of (top-left) M / beta.
        - If the site is exactly unitary (no dilation required) then C is diagonal
          with shape (k0, k0) and contains the normalized singular values s_norm.
        - Otherwise C is the assembled dilation Su (Su = [[S, C],[C,-S]]) or an
          identity-embedded enlargement of it. The side length k_core equals one of:
            * k0 = lcm(m,n)        (unitary short-circuit, x==1)
            * 2*k0                 (canonical dilation, x==2)
            * x*k0                 (identity-embedded replication when pad_pow is used)
        In all cases C represents the dilation of M / beta (not M).
    Vh : ndarray
        Full Vh unitary from SVD (shape n_full x n_full).
    m, n : int
        Unfolded matrix row/column dimensions prior to any identity post-normalization padding.
    d, p : int
        Equalization factors satisfying m * d == n * p == lcm(m, n).
    x : int
        Core replication / embedding factor (1 for unitary short-circuit, typically 2 or pad_pow).
    beta : float
        Per-site scale factor (the maximum singular value for the site). C dilates M / beta;
        callers must multiply per-site betas to obtain the global Gamma used when realizing
        the full block-encoding/unitary.
    out_shape, in_shape : tuple[int,...]
        The per-axis shapes used to fold/unfold the original site tensor (after any pre-unfold pad).
    svals: np.ndarray
        Normalized singular values after identity padding

    Notes
    -----
    - SiteFactor is immutable: collect instances safely for global realization.
    - The top-left block produced by U @ (top-left-of-C) @ Vh equals M / beta (modulo any identity-embedding).
    - beta must be accumulated multiplicatively across sites (Gamma = prod beta_i) when assembling a full
      block-encoding. The returned SiteFactor does not apply the global Gamma.
    """

    U: np.ndarray  # (m_full, m_full)
    C: np.ndarray  # (k, k) dilated core with k = x + max(n, m)
    Vh: np.ndarray  # (n_full, n_full)
    m: int
    n: int
    d: int
    p: int
    x: int
    beta: float
    out_shape: Tuple[int, ...]
    in_shape: Tuple[int, ...]
    svals: np.ndarray

    def ops(self, unfold: bool = True):
        """Returns U, C, Vh; in tensor views if not unfold for wiring into the network."""
        return (
            (self.U, self.C, self.Vh)
            if unfold
            else (
                self.U.reshape((*self.out_shape, self.m)),
                self.C.reshape((self.x, self.d, self.m, self.x, self.p, self.n)),
                self.Vh.reshape((self.n, *self.in_shape)),
            )
        )


class SiteLayer:
    """
    Per-site helper that computes an SVD and builds a normalized dilated core.

    Responsibilities
    - Given a site tensor T, factor() unfolds T using (out_axes, in_axes), computes a full SVD,
      and returns a SiteFactor that contains U, a normalized dilated core C, Vh and per-site metadata.
    - The returned C implements a unitary dilation of (unfolded M) / beta where beta is the
      per-site maximum singular value. The caller is responsible for accumulating per-site betas
      into a global scaling Gamma when realizing the full unitary/block-encoding.
    - Supports optional pre-unfold identity padding to powers of pad_pow to simplify later qubit mapping.

    pad_pow behavior
    - pad_pow is an integer > 1 or None.
      * If integer: T is identity-padded per axis up to the next power-of-pad_pow before unfolding.
      * If None: no pre-unfold identity padding; canonical dilation uses x==2 where applicable.

    Implementation notes
    - factor() separates two kinds of padding: "drop" (zero singulars added when m<n) and
      "identity" padding of normalized singulars (pads with ones to preserve top-left embedding).
    - Only pad_method == "identity" is implemented for identity-padding of singulars.
    """

    def __init__(self, T: np.ndarray, pad_pow: int = 2):
        self.T = np.asarray(T)
        self.pad_pow = None if pad_pow is None else int(pad_pow)

    def factor(
        self,
        out_axes: Sequence[int],
        in_axes: Sequence[int],
        tol: float = 1e-12,
        always_expand_core: bool = False,
        unitary_match_core: bool = False,
        pad_method: str = "identity",
    ) -> SiteFactor:
        """
        Factor this site and return a SiteFactor.

        Parameters
        ----------
        out_axes, in_axes : sequences of int
            Axes to treat as "output" and "input" when unfolding the site tensor; parsed via parse_axes.
        tol : float
            Numerical tolerance for near-unitarity / zero-scale detection.
        always_expand_core : bool
            If True, force building the dilated core even when the site is numerically unitary.
        unitary_match_core : bool
            If True, expand U and Vh (via Kronecker / identity-embed) to match the core size so that
            U, C, Vh are mutually compatible for direct multiplication without further reshaping.
        pad_method : {'identity'}
            Strategy to apply identity-padding to normalized singulars to reach canonical length k.
            Only 'identity' is supported (pads with ones on the unfolded-diagonal).

        Returns
        -------
        SiteFactor
            Immutable per-site factor with fields described in SiteFactor.__doc__.

        Semantics and important guarantees
        - The SiteFactor.C returned is the dilation for M / beta, where beta is SiteFactor.beta
          (per-site max singular). The caller must multiply per-site betas to form the global Gamma
          used to scale a full operator realization.
        - Two distinct paddings are used and intentionally separated:
            1) "drop" zero-padding of raw singulars when the unfolded M is short (m < n). These zeros
               represent truly dropped directions and are applied before normalization.
            2) "identity" padding of normalized singulars (pads with ones) to reach k = lcm(m,n),
               preserving the exact top-left embedding on padded directions.
        - If beta is numerically zero (<= tol) the site is treated as the zero map and beta is set to 0.0;
          the returned core dilates the zero map appropriately.
        - This method does not mutate the SiteLayer instance.
        """
        # Normalize axes
        out_axes, in_axes = parse_axes(self.T.ndim, out_axes, in_axes)
        # Normalize already to allow unscaled padding
        M_raw = _unfold_tensor(self.T, out_axes, in_axes)
        beta = float(np.linalg.svd(M_raw, compute_uv=False).max())
        if beta <= tol:
            beta = 0.0

        # Norm tensor
        T0 = np.zeros_like(self.T) if beta <= tol else self.T / beta
        # pad the site tensor
        T = (
            T0
            if self.pad_pow is None
            else identity_pad_to_power(
                T0,
                self.pad_pow,
                out_axes,
                in_axes,
            )
        )
        # store out/in shapes so consumer can reshape back
        out_shape = tuple(T.shape[i] for i in out_axes)
        in_shape = tuple(T.shape[i] for i in in_axes)

        # unfold to matrix M (rows=out, cols=in)
        M = _unfold_tensor(T, out_axes, in_axes)
        m, n = M.shape
        # canonical auxiliary dims
        k = math.lcm(m, n)
        d, p = k // m, k // n

        # full SVD
        U_full, s, Vh_full = np.linalg.svd(M, full_matrices=True)

        # Include hidden drop values
        if m < n:
            s = np.pad(s, (0, n - m))

        # Allow drift in tol to extract unitary core
        is_unitary = False
        if np.all(1.0 - s <= tol):
            s = np.ones_like(s)
            is_unitary = True

        # compute diagonal dilation s and complementary c (unitary_diag_dilation expects raw s)
        s_normed, c_vec, beta_norm = unitary_diag_dilation(s, gamma=None, tol=tol)
        beta *= beta_norm

        # pad normalized singulars and complement to length k according to pad_method
        if pad_method == "identity":
            pad_len = max(0, k - len(s))
            s_pad = np.pad(s_normed, (0, pad_len), constant_values=1.0)
            c_pad = np.pad(c_vec, (0, pad_len))
        else:
            raise ValueError("Only identity padding support")

        # If core is effectively unitary and caller does not force expansion, no dilution needed
        if not always_expand_core and is_unitary:
            Su = np.diag(s_pad)
            x = 1
        else:
            # Build the dilated core Su = [[S, C], [C, -S]]
            S_mat = np.diag(s_pad)
            Cdiag = np.diag(c_pad)
            Su = np.block([[S_mat, Cdiag], [Cdiag, -S_mat]])
            x = 2 if self.pad_pow is None else self.pad_pow

            # If x>2, embed Su into larger identity-padded block
            if x > 2:
                Su = identity_embed_pad(Su, (x * k,) * 2, (0,), (1,))

        # Optionally expand/match unitaries to core size
        if unitary_match_core:
            U_mat = np.kron(np.eye(d, dtype=complex), U_full)
            Vh_mat = np.kron(np.eye(p, dtype=complex), Vh_full)
            U_mat = identity_embed_pad(U_mat, (x * k,) * 2, (0,), (1,))
            Vh_mat = identity_embed_pad(Vh_mat, (x * k,) * 2, (0,), (1,))
        else:
            U_mat, Vh_mat = U_full, Vh_full

        return SiteFactor(
            U=U_mat,
            C=Su,
            Vh=Vh_mat,
            m=m,
            n=n,
            d=d,
            p=p,
            x=x,
            beta=beta,
            out_shape=out_shape,
            in_shape=in_shape,
            svals=s_pad,
        )


# -------------------------
# Registry Bookkeeping helper
# -------------------------


class AtomicRegistry:
    """
    Atomic register allocator — readable, minimal, deterministic.

    Ordering and invariants
    - reg_alocs[key] stores a tuple[int,...] in INTERNAL (MSB-first) order.
    - get(*keys) returns the concatenation of per-key tuples in LSB-first order (i.e., reversed per-key).
    - free_list is ascending; _assign takes from the smallest available atoms to keep allocations compact.
    - The physical site qudit is at the LSB atom position for that site.

    API behavior summary
    - set(key, size, overwrite=False) -> tuple (public LSB-first): allocates base_count(size) atoms for key.
    - get(*keys, ignore_missing=False) -> tuple (public LSB-first): returns concatenated atoms; KeyError if missing.
    - free(*keys, ignore_missing=False) -> None: frees keys and returns atoms to free_list; KeyError if any missing.
    - redistribute(src_keys, k, dst_dims, ...) : collects atoms (internal order) from src_keys and reassigns contiguous
      MSB-first chunks to destination keys (k,j) according to dst_dims (sizes converted via base_count).
    - ignore_missing can be thought of as treating missing keys as empty registers ().

    Example:
    regs = AtomicRegistry(atom_size=2)
    regs.set((0, -1), 4)   # allocates base_count(4)=2 atoms -> reg_alocs[(0,-1)] internal (MSB-first)
    regs.get((0, -1))      # returns tuple in LSB-first (least-significant atom first)
    """

    def __init__(self, atom_size: int = 2):
        self.size = 0
        self.atom_size = int(atom_size)
        if self.atom_size < 2:
            raise ValueError(f"atom_size must be >= 2, got: {self.atom_size}")
        self.free_list: List[int] = []
        self.reg_alocs: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def _norm_key(self, key: Tuple[int, int]) -> Tuple[int, int]:
        return (int(key[0]), int(key[1]))

    def _grow(self, n: int) -> None:
        """Grow the global atom pool by n new atoms."""
        self.free_list = (
            list(reversed(range(self.size, self.size + n))) + self.free_list
        )
        self.size += n

    def _assign(self, count: int) -> Tuple[int, ...]:
        """
        Take `count` atoms from the free pool (growing if needed), return them as a tuple.
        Deterministic: always pops from the front of free_list.
        """
        if count <= 0:
            return tuple()
        if len(self.free_list) < count:
            self._grow(count - len(self.free_list))
        self.free_list, out = self.free_list[:-count], tuple(self.free_list[-count:])
        return out

    def _get(self, *keys: Tuple[int, int]):
        return tuple(chain.from_iterable(self.reg_alocs.get(k, ()) for k in keys))

    def _free(self, keys: Set[Tuple[int, int]]):
        self.free_list = sorted(list(self._get(*keys)) + self.free_list, reverse=True)
        self.reg_alocs = {k: v for k, v in self.reg_alocs.items() if k not in keys}

    def set(
        self, key: Tuple[int, int], size: int, overwrite: bool = False
    ) -> Tuple[int, ...]:
        """
        Allocate and assign the register `key` to hold `base_count(size, atom_size)` atoms.

        If key already present:

          - overwrite=False (default) raises KeyError (stingy behavior).
          - overwrite=True frees previous atoms and assigns a fresh allocation (previous behavior).

        Returns the assigned tuple of atom indices for convenience.
        """
        k = self._norm_key(key)
        count = base_count(int(size), self.atom_size)

        if k in self.reg_alocs:
            if not overwrite:
                raise KeyError(f"Key already present: {k}")
            # explicit overwrite requested: free old atoms first
            self.free(k)

        self.reg_alocs[k] = self._assign(count)
        return self.get(k)

    def get(
        self, *keys: Tuple[int, int], ignore_missing: bool = False
    ) -> Tuple[int, ...]:
        """
        Return the concatenation of atom tuples for the requested keys in the order provided.
        Individual registers are reversed to fit LSB convention.

        Strict: raises KeyError if any key is not present.
        """
        keys = tuple(self._norm_key(k) for k in keys)

        if not ignore_missing and (missing := (set(keys) - self.reg_alocs.keys())):
            raise KeyError(f"Requested key(s) not present: {missing}")

        return self._get(*keys)

    def free(self, *keys: Tuple[int, int], ignore_missing: bool = False) -> None:
        """
        Free and remove the provided keys. By default missing keys raise KeyError.
        Set ignore_missing=True to silently ignore missing keys.

        Freed atoms are appended to the free_list (available for reallocation).
        """
        keys = tuple(self._norm_key(k) for k in keys)
        Skeys = set(keys)

        if not ignore_missing and len(Skeys) != len(keys):
            raise ValueError(f"Double free key {keys}")
        if not ignore_missing and (missing := Skeys - self.reg_alocs.keys()):
            raise KeyError(f"Missing keys to free: {missing}")

        self._free(Skeys & self.reg_alocs.keys())

    def redistribute(
        self,
        src_keys: Sequence[Tuple[int, int]],
        k: int,
        dest_dims: Sequence[Tuple[int, int]],
        ignore_missing: bool = False,
    ) -> None:
        """
        Move atoms from src_keys into destination keys (k, j) for each (j, size) in dst_dims.

        Semantics:
        - src_keys must be unique.
        - dst keys (k,j) must be unique.
        - A destination key may only be assigned if it is a source key (it will be collected and reassigned).
        - The total number of collected atoms (respecting ignore_missing) must equal needed atoms.
        """
        k = int(k)
        # convert dst sizes to atom counts
        dest_dims = [(int(j), base_count(s, self.atom_size)) for j, s in dest_dims]

        src_keys = [self._norm_key(s) for s in src_keys]
        S_src = set(src_keys)
        if len(S_src) != len(src_keys):
            raise ValueError("Duplicate source key not allowed for redistribute!")

        dest_keys = [(k, j) for j, _ in dest_dims]
        S_dest = set(dest_keys)
        if len(S_dest) != len(dest_keys):
            raise ValueError("Duplicate destination key not allowed for redistribute!")

        if overwrite := (S_dest & set(self.reg_alocs.keys())) - S_src:
            raise KeyError(f"Redistribute can only overwrite dynamically: {overwrite}")

        reg = self.get(*src_keys, ignore_missing=ignore_missing)
        # Memory is assigned precisely
        if len(reg) != sum(s for _, s in dest_dims):
            raise ValueError("Redistribute found size mismatch!")

        # Clear old keys
        for s in src_keys:
            self.reg_alocs.pop(s, None)
        # Distribute reg along new owners in order
        for j, s in dest_dims:
            self.reg_alocs[(k, j)], reg = reg[:s], reg[s:]


# -----------------------------------------------------------------------------
# TensorNetworkEncoding: general graph-based container
# -----------------------------------------------------------------------------


class TensorNetworkEncoding:
    """
    General graph/tensor-network container.

    Responsibilities:
      - Accept arbitrary graph description (nodes, edges, node-to-axes map).
      - Provide algorithm to build operators from grpah structure.
      - Produce per-node SiteLayer-like dilations for each node.
      - Provide local SVD growth heuristic routine (see your notes: greedy growth).
      - For now, just allows one to build Matrix from it.
      - greedy takes notes one by one and chooses which has the most input - output. So temp gets increased the least.
      - Should be structured to allow quantum circuit export.
      - Getting needed memory is easy. init to 0. if one needs some and not enough available,
        just set 0 and take anyway, returns once unused, this leaves the maximum needed memory at the end.

    Note:
      The Network is mostly assumed to have no free axis, except for the in- and output dimensions for each site.
      By merging dimensions, this is no proper restriction, except for the in- and ouput having to match.
      SiteLayer allows pad_pow = None, but this is not implemented for the Blockencoding.
      Empty dimensions (= 0) not allowed.
    """

    def __init__(
        self,
        V: List[Any],
        E: np.ndarray,
        pad_pow: int = 2,
    ):
        """
        V: ordered list of SiteLayer or raw tensors (np.ndarray). Raw tensors are wrapped.
        E: (n,n) int array, -1 = no edge, else axis index on row node used for edge to column node
          So edge from i to j from dimensions di to dj has E[i, j] == di and E[j, i] == dj.
        pad_pow: integer pad power of atomic dimension (states of qudit).
        """
        self.pad_pow = int(pad_pow)
        self.V = [
            v if isinstance(v, SiteLayer) else SiteLayer(np.asarray(v), self.pad_pow)
            for v in V
        ]
        self.E = np.asarray(E, dtype=int)
        self._validate_and_index()
        self._build_neighbors()

    def _validate_and_index(self) -> None:
        """
        Checks that all conections are valid, back and forth with matching dimension and only once.
        Reserved dimensions are unused, and optinally all non-reserved are used.
        All dimensions are non-negative or -1 if unused.
        Dimensions are > 0, since interpretation depends on context.
        NOTE: Diagonal is unused. Setting -1 for consistency.
        """
        n = len(self.V)
        if self.E.shape != (n, n):
            raise ValueError("E must be square with size == len(V)")

        used_axes = [set() for _ in self.V]
        for i in range(n):
            if self.E[i, i] != -1:
                raise ValueError(f"Loops not allowed, found on {i}")

            for j in range(i + 1, n):
                ai, aj = self.E[i, j], self.E[j, i]
                if (ai >= 0) ^ (aj >= 0):
                    raise ValueError(f"Half-edge between {i} and {j} not allowed")
                if ai == -1 and aj == -1:
                    continue
                if ai < -1 or aj < -1:
                    raise ValueError(
                        f"Invalid negative axis {ai} or {aj} found on ({i}, {j})"
                    )
                di, dj = (
                    self.V[i].T.shape[ai],
                    self.V[j].T.shape[aj],
                )  # Will error if ai / aj too large

                if di != dj:
                    raise ValueError(
                        f"Bond dim mismatch V[{i}].axis{ai}={di} vs V[{j}].axis{aj}={dj}"
                    )
                if ai in used_axes[i] or aj in used_axes[j]:
                    raise ValueError(f"Axis reuse: V[{i}].axis{ai} or V[{j}].axis{aj}")
                used_axes[i].add(ai)
                used_axes[j].add(aj)

            # i is now fully build
            # Enforce last two axes are physical; edges must not reference them
            dim = self.V[i].T.ndim - 2
            if any(ax >= dim for ax in used_axes[i]):
                raise ValueError(f"Reserved axis used on node {i}")

            # Enforce that physical dims are same
            if self.V[i].T.shape[-2] != self.V[i].T.shape[-1]:
                raise ValueError(
                    f"Site {i} physical dims must match: got {self.V[i].T.shape[-2:]}"
                )
            # Enforce that physical dims power of pad_pow
            if self.V[i].T.shape[-1] != next_pow(self.V[i].T.shape[-1], self.pad_pow):
                raise ValueError(
                    f"Site {i} physical dims must be power of pad_pow={self.pad_pow}: got {self.V[i].T.shape[-1]}"
                )
            # Enforce that all non-phys axes are used
            if len(used_axes[i]) < dim:
                raise ValueError(
                    f"Unused axis on node {i}, {[j for j in range(dim) if j not in used_axes[i]]}"
                )

            if any(s == 0 for s in self.V[i].T.shape):
                raise ValueError(
                    f"Zero axes on node {i}, {[j for j in range(dim) if self.V[i].T.shape[j] == 0]}"
                )

    def _build_neighbors(self) -> None:
        self.N = [
            {int(j) for j, val in enumerate(self.E[i]) if val >= 0}
            for i in range(len(self.V))
        ]

    @staticmethod
    def mpo_to_tensor_network(
        cores: List[np.ndarray],
        lc: np.ndarray = None,
        rc: np.ndarray = None,
        pad_pow: int = 2,
    ):
        """
        Convenience: convert open-chain MPO cores -> TensorNetworkEncoding.

        Args:
            cores : sequence of ndarray
                MPO cores in convention (Dl, Dr, phys_out, phys_in).
            lc : array_like or None
                Left boundary vector of length Dl_first. If provided, its entries are
                multiplied into the first core (preserves the left bond axis).
            rc : array_like or None
                Right boundary vector of length Dr_last. If provided, its entries are
                multiplied into the last core (preserves the right bond axis).
            pad_pow : int
                Passed through to TensorNetworkEncoding; if None the TNE will be built
                with pad_pow=None (whatever your TNE supports).

        Returns:
            TensorNetworkEncoding built from the MPO cores with lc/rc absorbed.
        """
        # Build node list, absorbing boundary vectors by multiplication (not contraction)
        V = [C.copy() for C in cores]
        n = len(V)
        E = np.full((n, n), -1, dtype=int)
        if n == 0:
            raise ValueError("Empty MPO not suported")

        if lc is None:
            lc = np.eye(1, V[0].shape[0], dtype=complex)[0]
        if rc is None:
            rc = np.eye(1, V[-1].shape[1], dtype=complex)[0]

        # validate 4-d cores
        for i, C in enumerate(V):
            if C.ndim != 4:
                raise ValueError(
                    "mpo_to_tensor_network expects 4-d MPO cores (Dl,Dr,phys_out,phys_in)"
                )

        if n == 1:
            V[0] = np.tensordot(
                lc, np.tensordot(V[0], rc, axes=([1], [0])), axes=([0], [0])
            )
        else:
            V[0] = np.tensordot(lc, V[0], axes=([0], [0]))
            V[-1] = np.tensordot(V[-1], rc, axes=([1], [0]))

            # first core had its left bond consumed: its remaining bond axis to node 1 is axis 0
            E[0, 1] = 0
            E[1, 0] = 0

            # internal links: node i has (Dl,Dr,phys_out,phys_in) so connect Dr (axis 1) -> Dl (axis 0) of next
            for i in range(1, n - 1):
                E[i, i + 1] = 1
                E[i + 1, i] = 0

        # Return convenience TensorNetworkEncoding (it will validate shapes/axes)
        return TensorNetworkEncoding(V, E, pad_pow=pad_pow)

    def next_site(self, S: set[int]) -> int:
        """
        Choose node in B with maximum total dimension in A minus total dimension in B, lower index if tied.
        A and B are sets of node indices.
        """

        def score(i: int) -> int:
            qs = self.V[i].T.shape
            A_connect = math.prod([qs[self.E[i, j]] for j in (self.N[i] - S)])
            B_connect = math.prod([qs[self.E[i, j]] for j in (self.N[i] & S)])
            return B_connect - A_connect

        # iterate sorted(B) so ties pick the smaller node id
        return min(sorted(S), key=score)

    def build_operator_sequence(
        self,
        unfold: bool = True,
        tol: float = 1e-12,
        diluted_cores: bool = False,
    ) -> Tuple[List[Tuple[np.ndarray, Tuple[int, ...]]], int, Tuple[float, ...]]:
        """
        Two-pass build:
        - Pass 1: sweep and factor each visited site -> store FactoredSite + local metadata.
        - Pass 2: realize (dilate / build_tensors) each FactoredSite with gamma, drive registry,
            and emit ops list of (tensor, target_registers).

        Returns:
        ops: list of (ndarray, tuple_of_register_indices)
        total_atoms: int registry size (number of atomic slots)
        gamma: float global rescaling factor (product of per-site betas)
        if post_prob is set, also returns statistic on post-selection probability
        """
        # processing sets
        n = len(self.V)
        remain = set(range(n))
        regs = AtomicRegistry(self.pad_pow)
        Ops, gamma, site_stats = [], 1.0, []

        # Place sites at the very front in order LSB
        for j in range(n):
            regs.set((j, self.V[j].T.ndim - 1), self.V[j].T.shape[-1])

        while remain:
            # Pick any node to process next. Default greedy tries to reduces ancilla use.
            q = self.next_site(remain)

            # determine local in/out neighbors and corresponding axes
            input = sorted(self.N[q] - remain)
            output = sorted(self.N[q] & remain)
            site = int(self.V[q].T.ndim) - 1

            # local axis indices lists for direction of Site, qudit at end as LSB
            in_axes = list(self.E[q, input]) + [site]  # in site
            out_axes = list(self.E[q, output]) + [site - 1]  # out site
            from_axes = list(self.E[input, q])

            # factorize site and dilation dimension
            op = self.V[q].factor(out_axes, in_axes, tol=tol)

            # redistribute registers for Vh (collect input registers + site input) into single dimension -1
            regs.redistribute(
                list(zip(input, from_axes)) + [(q, site)], q, [(-1, op.n)]
            )
            Vh_targ = regs.get((q, -1))

            # Allocate core and padding as neeed, core must be post-selected, axes == -2
            pad = regs.set((q, -3), op.p)
            C_targ = regs.set((q, -2), op.x) + pad + Vh_targ

            # Reinterpret into drop and input for U
            regs.redistribute(((q, -3), (q, -1)), q, ((-3, op.d), (-1, op.m)))
            regs.free((q, -3))  # Dropping padding dimensions
            U_targ = regs.get((q, -1))

            # finally redistribute the site register into outgoing bond registers
            regs.redistribute(
                [(q, -1)],
                q,
                [(j, self.V[q].T.shape[j]) for j in out_axes],
            )
            # Append operator and their targets
            U, C, Vh = op.ops(unfold)
            Ops += [(Vh, Vh_targ), (C, C_targ), (U, U_targ)]
            gamma *= op.beta
            site_stats.append(op.svals)

            remain.remove(q)

        # Check consistency of register, only out sites and core (axes = -2) left
        phys_keys = {(j, self.V[j].T.ndim - 2) for j in range(n)}
        present_keys = set(regs.reg_alocs.keys())
        postsel_keys = set(k for k in present_keys if k[1] == -2)

        # One post-select per site, can be empty though
        if len(postsel_keys) != n:
            raise RuntimeError(f"Not one Post-Select per site {postsel_keys}")

        # Unexpected keys (anything not a phys key or a post-selection key)
        if unexpected := present_keys - phys_keys - postsel_keys:
            raise RuntimeError(
                f"Unexpected registers remaining after sweep: {unexpected}"
            )

        # Ensure every site physical register remains
        if missing_phys := phys_keys - present_keys:
            raise RuntimeError(
                f"Missing physical site registers after sweep: {missing_phys}"
            )

        assert tuple(
            range(
                sum(
                    base_count(int(self.V[j].T.shape[-1]), self.pad_pow)
                    for j in range(n)
                )
            )
        ) == tuple(
            chain.from_iterable(
                reversed(regs.get((j, self.V[j].T.ndim - 2))) for j in range(n)
            )
        ), f"Qubits got scrambled: {tuple(regs.get((j, self.V[j].T.ndim - 2)) for j in range(n))}"

        if diluted_cores:
            return Ops, regs.size, gamma, site_stats
        else:
            return Ops, regs.size, gamma

    def build_dense_matrix(
        self, max_dims: int = 1 << 16, tol: float = 1e-12, post_prob: bool = False
    ):
        """
        Build full dense global operator by embedding each site operator into the global space and
        multiplying in sequence (operators returned by build_operator_sequence are applied left-to-right).
        """
        if post_prob:
            ops, nq, gamma, cores = self.build_operator_sequence(
                unfold=True, tol=tol, diluted_cores=post_prob
            )
            prob = float(np.prod(sum(c**2) / len(c) for c in cores))
        else:
            ops, nq, gamma = self.build_operator_sequence(
                unfold=True, tol=tol, diluted_cores=post_prob
            )
        total = int(self.pad_pow**nq)
        if total > max_dims:
            raise MemoryError(f"Total dimension {total} > max allowed {max_dims}")

        # initialize global operator as identity tensor of shape (out_dims..., in_dims...)
        split_dims = (self.pad_pow,) * (2 * nq)
        G = np.eye(total, dtype=complex).reshape(split_dims)

        # for Q, targets in ops:
        for Q, targets in ops:
            # permute operator axes and flatten to apply operation
            # target j means that output bit j in LSB is to be targeted
            op_axes = [nq - 1 - t for t in targets]
            perm = op_axes + [j for j in range(2 * nq) if j not in op_axes]
            G = Q @ G.transpose(perm).reshape((self.pad_pow ** len(targets), -1))

            # reshape back to tensor and invert permutation
            G = G.reshape(split_dims).transpose(np.argsort(perm))
        if post_prob:
            return G.reshape(total, total), gamma, prob
        else:
            return G.reshape(total, total), gamma

    def brute_contract(self):
        """
        Instance contractor: contract self.V / self.E into the dense operator H.
        Uses the deterministic neighbor-ordering by site index.
        Returns dense matrix with outputs then inputs in LSB site order.
        """
        n = len(self.V)
        # Tags per site: (target_site, target_axis_index) for each local axis, so where they connect
        # with physical out/in appended as (-2,site)/(-1,site).
        tags_per_site = tuple(
            tuple(
                (j, int(a))
                for _, j, a in sorted(
                    (
                        (self.E[i, j], j, self.E[j, i])
                        for j in range(n)
                        if self.E[j, i] >= 0
                    ),
                    key=lambda t: t[0],
                )
            )
            + ((-2, i), (-1, i))
            for i in range(n)
        )
        # Inititalize accumulator. 1-qudit singleton with trivial in- and output
        acc = np.ones((1, 1), dtype=complex)
        acc_tags = [(-2, -1), (-1, -1)]
        remaining = set(range(n))

        while remaining:
            s = self.next_site(remaining)

            # positions in acc_tags that target site s
            acc_dims = [j for j, tag in enumerate(acc_tags) if tag[0] == s]
            s_dims = [acc_tags[j][1] for j in acc_dims]

            # contract acc with site s over the matching dimensions
            acc = np.tensordot(acc, self.V[s].T, (acc_dims, s_dims))

            # remove consumed acc tags (targeting s) and append the remaining site tags
            acc_tags = [tag for tag in acc_tags if tag[0] != s] + [
                tag for j, tag in enumerate(tags_per_site[s]) if j not in s_dims
            ]
            remaining.remove(s)

        assert not tuple(
            tag for tag in acc_tags if tag[0] not in {-2, -1}
        ), f"Unexpected free dimensions after contraction {acc_tags}"

        # collect output/in positions and sort digits so LSB ordering matches other code
        out_pos = sorted(
            ((pos, tag[1]) for pos, tag in enumerate(acc_tags) if tag[0] == -2),
            key=lambda x: -x[1],
        )
        in_pos = sorted(
            ((pos, tag[1]) for pos, tag in enumerate(acc_tags) if tag[0] == -1),
            key=lambda x: -x[1],
        )
        assert (
            len(out_pos) == len(in_pos) == n + 1  # n sites and dummy
        ), f"Site dimensions got lost or added: {out_pos} x {in_pos}"
        acc = acc.transpose(tuple(p for p, _ in out_pos + in_pos))
        assert math.prod(acc.shape[:n]) == math.prod(
            acc.shape[n:]
        ), f"Resulting matrix not square! {math.prod(acc.shape[:n]), math.prod(acc.shape[n:])}"
        return acc.reshape((math.prod(acc.shape[:n]), -1))


# -------------------------
# QUBO builder
# -------------------------


I2 = np.eye(2, dtype=complex)
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
O2 = np.zeros((2, 2), dtype=complex)
P2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)


def QUBO_to_ising_parts(
    Q: np.ndarray, c: complex = 0, tol: float = 1e-12
) -> Tuple[complex, List[complex], Dict[Tuple[int, int], complex]]:
    """
    Convert a QUBO matrix Q (numpy square) into a list of linear terms and dict for quadratic
    terms with (i, j) : a_ij

    For a numpy Q (n x n) we use x = (1 - z)/2 mapping and expand:
      x_i x_j = (1 - z_i - z_j + z_i z_j) / 4
      x_i   = (1 - z_i) / 2
    The returned strings use 'Z' where z acts, 'I' otherwise.
    Indexing convention: qubit 0 is the rightmost character of the string.
    """
    Q = np.asarray(Q, dtype=complex)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("QUBO_to_ising expects a square numpy array")

    n = Q.shape[0]
    constant = complex(c)
    linear = [0.0] * n
    square = {}

    # diagonal contributions: Qii * x_i with x_i=(1-z_i)/2
    for i in range(n):
        q = Q[i, i] / 2.0
        constant += q
        linear[i] -= q

        # off-diagonals: for i<j treat q = Qsym[i,j] as coefficient of x_i x_j
        for j in range(i + 1, n):
            # x_i x_j = (1 - z_i - z_j + z_i z_j) / 4
            q = (Q[i, j] + Q[j, i]) / 4.0
            if abs(q) > tol:
                constant += q
                linear[i] -= q
                linear[j] -= q
                square[(i, j)] = q

    return (
        complex(constant),
        [complex(lin) for lin in linear],
        {k: complex(q) for k, q in square.items()},
    )


def QUBO_to_ising(
    Q: np.ndarray, c: complex = 0, tol: float = 0
) -> Tuple[List[Tuple[str, complex]], complex]:
    """
    Convert a QUBO matrix Q (numpy square) into a list with pairs (pauli_string, coeff).

    For a numpy Q (n x n) we use x = (1 - z)/2 mapping and expand:
      x_i x_j = (1 - z_i - z_j + z_i z_j) / 4
      x_i   = (1 - z_i) / 2
    The returned strings use 'Z' where z acts, 'I' otherwise.
    Indexing convention: qubit 0 is the rightmost (least-significant) character of the string.
    """
    offset, linear, square = QUBO_to_ising_parts(Q, c, tol)

    # helper to set string with Z on given qubit indices (qubit 0 is rightmost)
    n = len(linear)

    def z_string(indices):
        s = ["I"] * n
        for i in indices:
            s[n - 1 - i] = "Z"
        return "".join(s)

    # Filter out zero coefficients (numerical noise)
    return [(z_string([j]), a) for j, a in enumerate(linear) if abs(a) > tol] + [
        (z_string(p), a) for p, a in square.items() if abs(a) > tol
    ], offset


def QUBOSweepEncoding(Q: np.ndarray, c: complex = 0.0, tol: float = 1e-12):
    """
    Deterministic left->right sweep MPO encoder for a QUBO matrix Q (nxn).
    Produces an MPO (list of cores in (Dl,Dr,phys_out,phys_in) convention) and
    returns a TensorNetworkEncoding built from those cores.

    Construction (implements Sec. 6.1.1 of the paper):
      - Convert Q into Ising Pauli-Z expansion via to_ising.
      - Collect linear terms l_i and quadratic pairs (i,j,alpha) with i<j.
      - Sweep left-to-right allocating a slot for each pending (i,j) with i <= t < j.
      - Slot semantics (standard MPO trick):
          * At origin site i: place Z at W[-1, slot+1] (push Z into bond slot).
          * At target site j: place alpha * Z at W[slot+1, 0] (pull and weight).
      - Exception is first site that get constant term as well in upper left block.
      - Overall dimension D = s_max + 2 where s_max is maximum number of simultaneously
        pending pairs during the sweep.
      - lc has to carry the partial system and start collecting the linear term
        So one at 0 and D - 1.
      - rc simply reduces to the system, one at zero.
    """
    # obtain Pauli expansion and offset
    n = Q.shape[0]
    if not n:
        return TensorNetworkEncoding([], np.zeros((0, 0), dtype=complex))
    offset, L, square = QUBO_to_ising_parts(Q, c, tol)

    # Compute last index where site is needed, at least itself
    Last = list(range(n))
    ops = [set() for _ in range(n)]  # Stores (i, alpha_ij) for alpha_ij with i < j
    for (i, j), a in square.items():
        ops[j].add((i, a))
        Last[i] = max(Last[i], j)

    # Compute maximum number of concurrently active pairs, count earlier sites that reach to current
    s_max = max(sum(1 for s in range(t) if t <= Last[s]) for t in range(n))

    # Last site where still needed, so safe to drop after.
    Drop = [set() for _ in range(n)]
    for i, j in enumerate(Last):
        if i != j:
            Drop[j].add(i)

    D = s_max + 2  # MPO bond dim
    # initialize empty cores: list length n, each core (D, D, 2, 2)
    # Looks like giant identity unfolded, so just fold giant identity.
    Gs = [
        np.eye(2 * D, dtype=complex).reshape((D, 2, D, 2)).transpose(0, 2, 1, 3)
        for _ in range(n)
    ]
    # Special treatment for first with constant term
    Gs[0][-1, 0] = offset * I2 + L[0] * Z2
    # Place linear terms at W[D-1, 0]
    for t in range(1, n):
        Gs[t][-1, 0] = L[t] * Z2

    # Simulate sweep to assign slots on the fly
    # free_slots holds currently-free slot indices; init so pop() returns smallest index
    free_slots = list(range(s_max))
    # Register of stored sites: slot -> origin or None if free
    X = [None for _ in free_slots]
    # Mapping of where site is stored
    M = [None for _ in range(n)]
    for t in range(n):
        for i, a in ops[t]:
            # Place at W[slot+1, 0] = alpha * Z
            Gs[t][M[i] + 1, 0] = a * Z2
        # Free all that are no longer needed: return their slot to free_slots and clear assignment
        for d in Drop[t]:
            free_slots.append(M[d])
            M[d] = None
        # Store site if later needed
        if t < Last[t]:
            # allocate smallest available free slot (pop returns smallest due to reversed init)
            M[t] = free_slots.pop()
            idx = M[t] + 1
            # Only clear slot diagonal if it was used before
            if X[M[t]] is not None:
                Gs[t][idx, idx] = O2
            X[M[t]] = t
            # Save in slot W[D-1, slot+1] = Z
            Gs[t][-1, idx] = Z2

    assert all(v is None for v in M), f"QUBOSweep: slot-owner M not cleared: {M}"
    assert set(free_slots) == set(range(s_max)), "QUBOSweep: Register not clear"

    return TensorNetworkEncoding.mpo_to_tensor_network(
        Gs,
        lc=np.eye(1, D, D - 1, dtype=complex)[0],
        rc=np.eye(1, D, dtype=complex)[0],
        pad_pow=2,
    )


def QUBOPauliSumEncoding(Q: np.ndarray, c: complex = 0.0, tol: float = 1e-12):
    """
    Pauli-sum tensor-sum encoding of a QUBO (sec. 6.1.2). Constructs a selector
    tensor and per-site blocks for each Pauli-string term.

    Returns a TensorNetworkEncoding implementing the tensor-sum construction
    Each physical site has a T x T x 2 x 2 tensor where row t corresponds to the ising
    string t. This matches the simple block-diagonal construction in the paper and is
    appropriate when the number of Pauli-strings L ~ O(n^2) is acceptable.
    The all I string is counted as well for the offset.

    NOTE Optional exact selector reduction (Gram-based):

    When the Pauli-sum selector dimension T (number of terms) is large but many terms are linearly
    dependent or strongly correlated, the minimal exact selector dimension r can be computed by forming
    the T×T Gram matrix G_{tu} = prod_i trace(L_{i,t}^† L_{i,u}), where L_{i,t} is the 2×2 local
    operator at site i for term t. Diagonalize G = U Σ U^† and keep only non-negligible eigenvalues (≥ tol).
    With W = U Σ^{-1/2} (T×r) the transformed per-site selector blocks are
    V'_i[α,β] = sum_k conj(W[k,α]) * W[k,β] * L_{i,k}

    The resulting per-site selector dimension is r (minimal and exact).
    This transformation is exact (no truncation) and often drastically reduces local selector/bond
    sizes when terms are redundant. It is cheaper than a full TT-SVD on the selector-chain.
    """
    n = Q.shape[0]
    if not n:
        if abs(c) > tol:
            raise ValueError(
                f"Not possible to encode non-zero constant without qubits: c ={c}"
            )
        return TensorNetworkEncoding([], np.zeros((0, 0), dtype=complex))
    terms, offset = QUBO_to_ising(Q, c, tol)
    # Add cosntant term if not neglectable or would be empty otherwise
    if abs(offset) > tol or not len(terms):
        terms = [("I" * n, offset)] + terms
    t = len(terms)

    V = [
        np.eye(2 * t, dtype=complex).reshape(2, t, 2, t).transpose(1, 3, 0, 2)
        for _ in range(n)
    ]

    for j, (s, a) in enumerate(terms):
        # s is length-n string; index conv: qubit 0 is rightmost (LSB)
        for k in [idx for idx, c in enumerate(s[::-1]) if c == "Z"]:
            V[k][j, j] = Z2
        V[0][j, j] *= a

    # Build edge connectivity
    E = -np.ones((n, n), dtype=int)
    if n == 1:
        V[0] = V[0].sum(axis=(0, 1))
    elif n == 2:
        # Multiedge not allowed, so we merge them
        V[0] = V[0].reshape(-1, 2, 2)
        V[1] = V[1].transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        E[0, 1] = E[1, 0] = 0
    else:
        for i in range(n):
            E[i, (i + 1) % n] = 1
            E[(i + 1) % n, i] = 0

    return TensorNetworkEncoding(V, E, pad_pow=2)


def QUBONetworkEncoding(
    Q: np.ndarray, c: complex = 0.0, kd: int = None, kc: int = None, tol: float = 1e-12
):
    """Graph-based per-term tensor-sum encoding of a QUBO.

    Construct an exact tensor-network representation of H(x)=x^T Q x + c by assembling
    per-term product tensors (constant, linear and quadratic). Each term is represented
    as a product of 2×2 local blocks (I or projector P) and summed via per-site selector
    indices. Terms with |coeff| <= tol are omitted.

    Key points / invariants
    - Per-site tensors: selector axes (one per neighbor, in ascending node order, skipping the
    local node) followed by two physical axes (out, in). This canonical ordering is preserved.
    - Exactness: the assembled TensorNetworkEncoding reproduces the QUBO operator up to
    floating-point error (and any singular-value truncation controlled by tol).

    Local rank-reduction primitive: triangle(i,j,k)
    - Purpose: locally refactor a 3-cycle into a chain to reduce internal selector bonds when
    local linear dependence exists.
    - Operation:
    1. contract site tensors i,j,k exactly into a single 3-way core,
    2. pick center m among {i,j,k} with the largest external selector footprint,
    3. perform two SVD splits to re-factor the core into a chain a--m--c,
    4. refold the three resulting cores back into per-site tensors, preserving canonical axis order.
    - Numerical rules: singular values <= tol are treated as zero; a minimal bond of 1 is kept
    to avoid degenerate zero-sized axes. The transform is algebraically exact up to the chosen tol.

    Limitations and usage notes
    - Purely local SVD moves (pairwise or triangle) reduce bonds only when local or global linear
    dependence exists. For dense/linearly independent term-sets no sequence of local moves will
    guarantee small constant bond-dimensions; a global selector change-of-basis (Gram reduction
    or TT-SVD over the selector index) is required for maximal compression.
    - Contracting three sites exactly can produce large temporaries (cost ∝ product of external
    selector dims × 2^3). Prefer triangles that include at least one low-footprint node.

    Returns
    - TensorNetworkEncoding: exact (up to tol) network representing the QUBO; triangle(...) is
    provided as a local primitive to be used by greedy reduction heuristics.

    NOTE: This is simply inferior to the PauliSum. The idea was to use the higher conenctivit,
    but the overhead is way to high. An alternative would be to force a different structure
    instead of MPO. Maybe a tree or a k-local graph.
    """
    Q, c = np.asarray(Q, dtype=complex), complex(c)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("QUBONetworkEncoding expects a square numpy array")
    n = Q.shape[0]
    if not n:
        if abs(c) > tol:
            raise ValueError(
                f"Not possible to encode non-zero constant without qubits: c = {c}"
            )
        return TensorNetworkEncoding([], np.zeros((0, 0), dtype=complex))

    kd = int(np.sqrt(n)) if kd is None else int(kd)
    kc = kd**2 if kc is None else int(kc)

    V = [np.eye(2, dtype=complex).reshape((1,) * (n - 1) + (2, 2)) for _ in range(n)]

    def add(i, U):
        """Add tensorsite U (2 x 2 at the last two dimensions) to site i"""
        v = np.pad(V[i], ((0, 1),) * (n - 1) + ((0, 0), (0, 0)))
        v[(-1,) * (n - 1) + (None, None)] += U
        V[i] = v

    def sel_axis(p, q):
        return q if q < p else q - 1

    def sel_site(p, q):
        return q if q < p else q + 1

    def split(M):
        U, s, V = np.linalg.svd(M)
        # binary search for significant part
        r, e = 0, len(s)
        while r < e:
            m = (r + e) // 2
            if s[m] > tol:
                r = m + 1
            else:
                e = m
        return U[:, :r], r, s[:r, None] * V[:r, :]

    def triangle(i, j, k):
        """
        Locally refactor a 3-cycle (i, j, k) into a chain using two SVD splits.

        Important axis semantics reminder:
        - Each site tensor V[s] uses the canonical ordering:

            (selector axes..., phys_out, phys_in).
          The last two axes are physical and must NOT be treated as selector axes.
        - sel_axis(p, q) / sel_site(p, q) convert between selector-axis index and

          site index, accounting for the missing local selector axis at position p.
          Use these helpers whenever you map between axis positions and site indices.
        - This routine unfolds only selector axes (explicitly referencing their indices)

          and therefore never touches the final two physical axes.

        Parameters
        ----------
        i, j, k : int
            Three distinct site indices forming a cycle to be factored.
        """
        assert (
            i != j != k != i
        ), f"triangle needs to be formed from three sites {i, j, k}"

        ij, ik, ji, jk, ki, kj = (
            sel_axis(i, j),
            sel_axis(i, k),
            sel_axis(j, i),
            sel_axis(j, k),
            sel_axis(k, i),
            sel_axis(k, j),
        )
        Si, Sj, Sk = (list(V[x].shape) for x in (i, j, k))
        Vi, Vj, Vk = (
            unfold_tensor(vx, (xy, xz)).reshape(vx.shape[xy], vx.shape[xz], -1)
            for vx, xy, xz in ((V[i], ij, ik), (V[j], ji, jk), (V[k], ki, kj))
        )
        T = np.einsum("abi,acj,bck->ijk", Vi, Vj, Vk)
        a, b, c = T.shape

        Vi, r1, T = split(T.reshape(a, b * c))
        Vj, r2, Vk = split(T.reshape(r1 * b, c))

        Si[ij], Si[ik] = r1, 1
        Sj[ji], Sj[jk] = r1, r2
        Sk[ki], Sk[kj] = 1, r2

        V[i], V[j], V[k] = (
            refold_tensor(Vi.T, Si, (ij, ik)),
            refold_tensor(Vj.reshape(r1, b, r2).transpose(0, 2, 1), Sj, (ji, jk)),
            refold_tensor(Vk, Sk, (ki, kj)),
        )

    def contract_greedy():
        """
        Greedy local triangle compression pass.

        Behavior summary:
        - For each site i, collect selector-axis sizes (i.e., all axes EXCEPT the final

          two physical axes) and sort them by descending size to find the largest 'offenders'.
        - We explicitly skip the physical (out/in) axes when building the offending-list D:

          V[i].shape[:-2] yields only selector axes. This avoids accidentally treating the
          physical axes as selector axes (a common source of bugs when using enumerate(V[i].shape)).
        - If the site has at least three selector axes and exceeds configured thresholds (kd/kc)

          we attempt a local triangle refactor using triangle(i, sel_site(i, ...), sel_site(i, ...)).
        """
        for i in range(n):
            while True:
                # Build list of dimensions and sort by biggest offender
                D = list(
                    sorted(
                        ((d, j) for j, d in enumerate(V[i].shape[:-2]) if d > 1),
                        key=lambda x: -x[0],
                    )
                )
                # Last two are free to go, else at most kd and at largest kc
                if len(D) < 3 or (len(D) <= kd and D[0][0] < kc):
                    break
                triangle(i, sel_site(i, D[1][1]), sel_site(i, D[0][1]))

    # Constant term is simply multiplied in
    V[0] *= c
    for j in range(n):
        # Linear terms, V[j] == q * P2
        q = Q[j, j]
        if abs(q) > tol:
            for k in range(n):
                add(k, q * P2 if k == j else I2)
            contract_greedy()

        # Quadratic terms, V[i] == a * P2, V[j] == P2
        for i in range(j):
            a = Q[i, j] + Q[j, i]
            if abs(a) > tol:
                for k in range(n):
                    add(k, a * P2 if k == i else P2 if k == j else I2)
                contract_greedy()

    # Using simply order of sites as order of dimensions, skipping loop
    # Build E with E[i,j] = -1 if i==j, else j if i>j, else j-1 (for i<j).
    E = np.tile(np.arange(n), (n, 1)) - (np.arange(n)[:, None] < np.arange(n))
    E[np.arange(n), np.arange(n)] = -1

    return TensorNetworkEncoding(V, E, pad_pow=2)


# -------------------------
# Dense Builders
# -------------------------


def qubo_direct_dense(Q: np.ndarray, c: complex = 0.0) -> np.ndarray:
    """
    Build the diagonal dense matrix H for QUBO E(x) = x^T Q x + c in the computational basis.

    Bit ordering: x[i] = bit at position (n-1-i) of integer b (matches the test helpers).
    Returns an (2^n x 2^n) complex ndarray with only diagonal entries filled.
    """
    Q = np.asarray(Q, dtype=complex)
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be a square (n,n) array")
    N = 1 << n
    H = np.zeros((N, N), dtype=complex)
    for b in range(N):
        x = np.array([(b >> i) & 1 for i in range(n)], dtype=complex)
        H[b, b] = c + x @ Q @ x
    return H


def brute_contract_mpo(Ms, lc, rc):
    """
    Build full dense operator H = sum_{i0..in} lc[i0] rc[in] kron_s Ms[s][:,:,i_s,i_{s+1}].
    Supports varying bond dims and avoids enumerating all chains at once.
    Ms: list of tensors shape (D_left,D_right,p,p)
    lc: shape (D_left_of_first,)
    rc: shape (D_right_of_last,)
    """
    n = len(Ms)
    if n == 0:
        return np.zeros((1, 1), dtype=complex)

    assert Ms[0].ndim == 4, "First MPO site wrong dimension"
    p = Ms[0].shape[2]
    # basic checks
    for s, M in enumerate(Ms):
        assert M.ndim == 4, f"MPO site with wrong dimension {s}"
        assert M.shape[2] == p and M.shape[3] == p, "physical dims mismatch"
        if s > 0:
            assert Ms[s - 1].shape[1] == M.shape[0], "bond dims mismatch between sites"

    lc = np.asarray(lc, dtype=complex)
    rc = np.asarray(rc, dtype=complex)
    assert lc.shape[0] == Ms[0].shape[0]
    assert rc.shape[0] == Ms[-1].shape[1]

    # Start by contracting left boundary into first site:
    # T shape -> (Dr, p, p)
    T = np.tensordot(lc, Ms[0], axes=([0], [0]))  # result (Dr, p, p)

    # Iteratively incorporate sites 1..n-1
    for s in range(1, n):
        # Contract next site -> (Dr, p, p, p^s, p^s)
        T = np.tensordot(Ms[s], T, axes=([0], [0]))
        # transpose in LSB order, so new dimensions are left and combine
        T = T.transpose(0, 1, 3, 2, 4).reshape(-1, p ** (s + 1), p ** (s + 1))

    # Finally contract right boundary
    H = np.tensordot(rc, T, axes=([0], [0]))  # shape (p^n, p^n)
    return H


# -------------------------
# DEBUG helpers
# -------------------------


class MonkeyPatch:
    """Temporarily replace obj.name with `new` and restore on exit.

    Usage:
      with MonkeyPatch(module_or_class, "funcname", new_callable):
          ... patched ...
    """

    def __init__(self, obj, name, new):
        self.obj = obj
        self.name = name
        self.new = new
        self._orig = None

    def __enter__(self):
        self._orig = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.new)
        return self

    def __exit__(self, exc_type, exc, tb):
        setattr(self.obj, self.name, self._orig)
        return False  # don't suppress exceptions


class Traced(MonkeyPatch):
    """Scoped tracer: wrap and install a simple IN/OUT tracer.

    Forms:
      with Traced(module_or_class, "name", tag="lbl"):     # patch obj.name
      with Traced(callable_obj, tag="lbl"):                # auto-resolve module.attr (module-level funcs)

    The wrapper captures the original callable at construction time and prints IN/OUT lines via `writer`.
    """

    def __init__(
        self,
        target,
        name: str = None,
        tag: str = None,
        repr_args: bool = True,
        writer=print,
    ):
        # resolve (obj, name) if caller passed a bare callable
        if name is None and callable(target):
            mod_name = getattr(target, "__module__", None)
            func_name = getattr(target, "__name__", None)
            if not mod_name or not func_name:
                raise ValueError(
                    "Can't auto-resolve callable; pass (obj, name) instead"
                )
            try:
                mod = __import__(mod_name, fromlist=[func_name])
            except Exception:
                raise ValueError(
                    f"Can't import module {mod_name}; pass (obj, name) instead"
                )
            if not hasattr(mod, func_name) or getattr(mod, func_name) is not target:
                raise ValueError("Auto-resolution failed; pass (obj, name) instead")
            obj, name = mod, func_name
        elif name is None:
            raise ValueError("When target is not callable, provide (obj, name)")
        else:
            obj = target

        orig = getattr(obj, name)
        tag = tag or getattr(orig, "__name__", "<call>")
        writer = writer
        repr_args = bool(repr_args)

        def wrapper(*args, **kwargs):
            if repr_args:
                try:
                    args_s = ", ".join(repr(a) for a in args)
                    kw_s = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                    writer(
                        f"[TRACE {tag}] IN: {args_s}{', ' if args and kwargs else ''}{kw_s}"
                    )
                except Exception:
                    writer(f"[TRACE {tag}] IN: (args suppressed)")
            else:
                writer(f"[TRACE {tag}] IN")
            out = orig(*args, **kwargs)
            try:
                writer(f"[TRACE {tag}] OUT: {out!r}")
            except Exception:
                writer(f"[TRACE {tag}] OUT: (result suppressed)")
            return out

        try:
            wrapper.__name__ = getattr(orig, "__name__", "wrapped")
        except Exception:
            pass

        # install the wrapper via MonkeyPatch
        super().__init__(obj, name, wrapper)


if __name__ == "__main__":

    def check_scale_and_report(
        A, B, nameA, nameB, tol_nz=1e-12, atol=1e-9, scale_tol=1e-9
    ):
        """
        Print nothing if A ~= B (within atol) or A ~= 1 * B.
        If there exists a scalar `scale` with A ~= scale * B:
        - print the scale only when |scale - 1| > scale_tol (i.e., scaling different from 1).

        Otherwise print a short linear-independence / mismatch message with residual.
        """
        da = np.asarray(A).ravel()
        db = np.asarray(B).ravel()

        # prefer an index where B is non-zero to avoid division by zero
        idxs_b = [i for i, v in enumerate(db) if abs(v) > tol_nz]
        idxs_a = [i for i, v in enumerate(da) if abs(v) > tol_nz]

        if idxs_b:
            idx = idxs_b[0]
            scale = da[idx] / db[idx]
            scaled = scale * db
            ok = np.allclose(da, scaled, atol=atol, rtol=0)
            if ok:
                # Only report when scale significantly differs from 1
                if abs(scale - 1) > scale_tol:
                    print(f"SCALE {nameA} vs {nameB}: scale = {scale}")
                # else: matched and scale ~= 1 -> print nothing
            else:
                print(f"LINEAR_INDEPENDENT {nameA} vs {nameB}")
        elif idxs_a:
            # B is zero but A has non-zero entries -> cannot form finite scale
            idx = idxs_a[0]
            print(f"LINEAR_INDEPENDENT {nameA} vs {nameB}")
        # else both effectively zero -> print nothing

    def is_unitary(U):
        n, m = U.shape
        return n == m and np.allclose(U @ U.conj().T, np.eye(n, dtype=complex))

    def analyze_qubo(Q, c):
        H_direct = qubo_direct_dense(Q, c=c)

        tne = enc(Q, c=c, tol=1e-12)
        # run encoder
        H_tne = tne.brute_contract()
        H_u_raw, g = tne.build_dense_matrix()
        H_u = g * H_u_raw[: 2 ** Q.shape[0], : 2 ** Q.shape[1]]

        d_direct = np.diag(H_direct)
        d_tne = np.diag(H_tne)
        d_u = np.diag(H_u)

        # Compare available matrices
        print(" H_direct diag:", np.round(d_direct, 6))
        print(" H_tne diag:   ", np.round(d_tne, 6))
        print(" H_u diag:     ", np.round(d_u, 6))
        print("Encoding is unitary?", is_unitary(H_u_raw), f"Scaling g: {g}")

        # check the three relevant pairs, print only on mismatch
        check_scale_and_report(d_direct, d_tne, "H_direct", "H_tne")
        check_scale_and_report(d_direct, d_u, "H_direct", "H_u")
        check_scale_and_report(d_tne, d_u, "H_tne", "H_u")

    rng = np.random.default_rng(123)

    Qs = [
        np.array([[1.0, 0.5], [0.5, -1.0 / 3.0]]),
        np.array([[0.0, 0.75], [0.75, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]

    for enc in (QUBOSweepEncoding, QUBOPauliSumEncoding, QUBONetworkEncoding)[2:]:
        for n in range(1, 5):
            print("\n=== Inspect encoder:", enc.__name__, "n=", n, "===")
            analyze_qubo(rng.normal(size=(n, n)), 0.123)
        for Q in Qs:
            print("\n=== Inspect encoder:", enc.__name__, "Q=\n", Q, "\n==")
            analyze_qubo(Q, 0.0)

    print("\nDIAG end.")
