
import pytest
import numpy as np
from itertools import product
from numpy.testing import assert_equal

from newlang.kernel import kernel, Group

@pytest.mark.parametrize("gsize", [(512,1,1),(511,1,1),(1,16,1),(1,1,16),(1,1,1)])
@pytest.mark.parametrize("lsize", [(64,1,1),(1,1,1)])
def test_group_iteration(gsize, lsize):
    def get_range(i):
        return range((gsize[i] + lsize[i] - 1) // lsize[i])

    def get_group_ranges():
        return product(get_range(0),get_range(1),get_range(2))

    res_ids = []
    res_offsets = []

    @kernel
    def test(gr):
        res_ids.append(gr.group_id())
        res_offsets.append(gr.work_offset())

    test(Group(gsize, lsize))
    exp_res_ids = [(i, j, k) for i, j, k in get_group_ranges()]
    assert(res_ids == exp_res_ids)
    exp_res_offsets = [(i * lsize[0], j * lsize[1], k * lsize[2]) for i, j, k in get_group_ranges()]
    assert_equal(res_offsets, exp_res_offsets)


def test_group_load_small():
    gsize = (8,1,1)
    lsize = (8,1,1)

    res = []

    @kernel
    def test(gr, arr):
        a = gr.load(arr[gr.work_offset()[0]:], shape=gr.group_shape()[0])
        res.append(a.compressed())

    src = np.arange(12)
    test(Group(gsize, lsize), src)
    expected = [np.array([0,1,2,3,4,5,6,7])]
    assert_equal(res, expected)

def test_group_load():
    gsize = (16,1,1)
    lsize = (8,1,1)

    res = []

    @kernel
    def test(gr, arr):
        a = gr.load(arr[gr.work_offset()[0]:], shape=gr.group_shape()[0])
        res.append(a.compressed())

    src = np.arange(12)
    test(Group(gsize, lsize), src)
    expected = [np.array([0,1,2,3,4,5,6,7]),np.array([8,9,10,11])]
    assert_equal(res, expected)


def test_group_store():
    gsize = (16,1,1)
    lsize = (8,1,1)

    @kernel
    def test(gr, arr1, arr2):
        gid = gr.work_offset()
        a = gr.load(arr1[gid[0]:], shape=gr.group_shape()[0])
        gr.store(arr2[gid[0]:], a)

    src = np.arange(12)
    dst = np.full(16, fill_value=-1)
    test(Group(gsize, lsize), src, dst)
    assert_equal(dst, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1])


def test_group_empty():
    gsize = (8,1,1)
    lsize = (8,1,1)

    res = []

    @kernel
    def test(gr):
        a = gr.empty((3,7), dtype=np.int32)
        assert_equal(a.shape, (3,7))

    test(Group(gsize, lsize))
