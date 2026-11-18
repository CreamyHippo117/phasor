"""Milestone 9: finite groups (the dihedral group is the exit demo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.domains import algebra_math as am


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_dihedral_groups_are_groups_of_order_2n(n):
    d = am.dihedral(n)
    d.verify()
    assert d.order == 2 * n and d.name == f"D{n}"
    assert not d.is_abelian()
    r, s = 1, n  # the generating rotation and a reflection
    assert d.element_order(r) == n and d.element_order(s) == 2
    assert all(d.element_order(n + k) == 2 for k in range(n))  # every reflection is its own inverse
    assert d.mul(s, r) == d.mul(d.inverse(r), s) and d.mul(r, s) != d.mul(s, r)  # r s = s r^-1
    assert d.generated([r, s]) == frozenset(range(2 * n))  # two generators give everything


def test_the_cayley_table_is_a_latin_square():
    d = am.dihedral(4)
    t = np.array(d.table)
    for k in range(d.order):
        assert sorted(t[k]) == list(range(d.order)) and sorted(t[:, k]) == list(range(d.order))
    assert d.labels[:4] == ("e", "r", "r2", "r3") and d.labels[4] == "s"


def test_dihedral_elements_really_act_as_rotations_and_reflections_of_the_polygon():
    n = 5
    d = am.dihedral(n)
    for a in range(d.order):
        m = am.action_matrix(n, a)
        assert abs(np.linalg.det(m) - (-1.0 if a >= n else 1.0)) < 1e-12
        for b in range(d.order):  # the matrices multiply like the group does
            assert np.allclose(am.action_matrix(n, d.mul(a, b)), m @ am.action_matrix(n, b)), (a, b)
    # a rotation by one step maps the first vertex onto the next
    v = np.array([0.0, 1.0])
    ang = am.action_matrix(n, 1) @ v
    assert math.atan2(ang[1], ang[0]) == pytest.approx(math.pi / 2 + 2 * math.pi / n)


def test_subgroups_lagrange_and_cosets():
    d = am.dihedral(4)
    subs = d.subgroups()
    assert len(subs) == 10  # the known count for D4
    assert all(d.order % len(h) == 0 for h in subs)  # Lagrange's theorem
    rotations = d.generated([1])
    assert len(rotations) == 4 and d.is_normal(rotations)  # index 2 is always normal
    cosets = d.left_cosets(rotations)
    assert len(cosets) == 2 and set().union(*cosets) == set(range(8))  # they partition the group
    assert not d.is_normal(d.generated([4]))  # a single reflection is not normal in D4
    assert d.is_normal(d.generated([2]))  # the centre {e, r2} is
    centre = [a for a in range(8) if all(d.mul(a, b) == d.mul(b, a) for b in range(8))]
    assert centre == [0, 2]


def test_cyclic_and_symmetric_groups():
    z = am.cyclic(12)
    z.verify()
    assert z.is_abelian() and z.element_order(5) == 12 and z.element_order(4) == 3
    assert sorted(len(h) for h in z.subgroups()) == [1, 2, 3, 4, 6, 12]  # one per divisor
    s3 = am.symmetric(3)
    s3.verify()
    assert s3.order == 6 and not s3.is_abelian()
    assert sorted(len(h) for h in s3.subgroups()) == [1, 2, 2, 2, 3, 6]
    assert sorted(len(h) for h in am.dihedral(3).subgroups()) == sorted(
        len(h) for h in s3.subgroups()
    )  # D3 = S3
    with pytest.raises(ValueError, match="n >= 3"):
        am.dihedral(2)
    with pytest.raises(ValueError, match="1 to 5"):
        am.symmetric(9)


def test_verify_catches_a_table_that_is_not_a_group():
    bad = am.Group("bad", ((0, 1, 2), (1, 2, 0), (2, 0, 0)), ("a", "b", "c"))
    with pytest.raises(ValueError, match="inverse|associative"):
        bad.verify()
