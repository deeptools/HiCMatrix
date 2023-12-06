import logging
import os.path
import sys
import warnings
# from past.builtins import zip
from collections import OrderedDict
from os import unlink
from tempfile import NamedTemporaryFile

import numpy as np
import numpy.testing as nt
import pytest
from intervaltree import Interval, IntervalTree
from scipy.sparse import coo_matrix, csr_matrix
from six import iteritems

from hicmatrix import HiCMatrix as hm

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")


def test_load_h5_save_and_load_cool():
    hic = hm.hiCMatrix(ROOT + 'Li_et_al_2015.h5')

    outfile = NamedTemporaryFile(suffix='.cool', prefix='hicexplorer_test')  # pylint: disable=R1732
    hic.matrixFileHandler = None
    hic.save(pMatrixName=outfile.name)

    hic_cool = hm.hiCMatrix(outfile.name)

    nt.assert_equal(hic_cool.matrix.data, hic.matrix.data)
    chrom_cool, start_cool, end_cool, _ = list(zip(*hic_cool.cut_intervals))
    chrom, start, end, _ = list(zip(*hic_cool.cut_intervals))

    nt.assert_equal(chrom_cool, chrom)
    nt.assert_equal(start_cool, start)
    nt.assert_equal(end_cool, end)


def test_load_h5_load_cool_weight():
    hic_h5 = hm.hiCMatrix(ROOT + 'Li_et_al_2015.h5')
    hic_cool = hm.hiCMatrix(ROOT + 'Li_et_al_2015.cool')

    # there is always a small gap due to rounding errors and inaccurate floating operations
    # test if it is equal for up to 10 decimal positions
    nt.assert_almost_equal(hic_cool.matrix.data, hic_h5.matrix.data, decimal=10)
    chrom_cool, start_cool, end_cool, _ = list(zip(*hic_cool.cut_intervals))
    chrom, start, end, _ = list(zip(*hic_cool.cut_intervals))

    nt.assert_equal(chrom_cool, chrom)
    nt.assert_equal(start_cool, start)
    nt.assert_equal(end_cool, end)


def test_load_h5_save_and_load_cool_2():
    hic = hm.hiCMatrix(ROOT + 'small_test_matrix.h5')

    outfile = NamedTemporaryFile(suffix='.cool', prefix='hicexplorer_test')  # pylint: disable=R1732
    hic.matrixFileHandler = None
    hic.save(pMatrixName=outfile.name)

    hic_cool = hm.hiCMatrix(outfile.name)

    nt.assert_equal(hic_cool.matrix.data, hic.matrix.data)
    chrom_cool, start_cool, end_cool, _ = list(zip(*hic_cool.cut_intervals))
    chrom, start, end, _ = list(zip(*hic_cool.cut_intervals))

    nt.assert_equal(chrom_cool, chrom)
    nt.assert_equal(start_cool, start)
    nt.assert_equal(end_cool, end)


def test_load_cool_save_and_load_h5():
    hic = hm.hiCMatrix(ROOT + 'Li_et_al_2015.cool')

    outfile = NamedTemporaryFile(suffix='.h5', prefix='hicexplorer_test')  # pylint: disable=R1732
    hic.matrixFileHandler = None
    hic.save(pMatrixName=outfile.name)

    hic_cool = hm.hiCMatrix(outfile.name)

    nt.assert_equal(hic_cool.matrix.data, hic.matrix.data)
    chrom_cool, start_cool, end_cool, _ = list(zip(*hic_cool.cut_intervals))
    chrom, start, end, _ = list(zip(*hic_cool.cut_intervals))

    nt.assert_equal(chrom_cool, chrom)
    nt.assert_equal(start_cool, start)
    nt.assert_equal(end_cool, end)


def test_save_load_cool():
    outfile = '/tmp/matrix.cool'
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]
    hic = hm.hiCMatrix()
    hic.nan_bins = []
    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    # make matrix symmetric
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.fillLowerTriangle()
    # hic.correction_factors = np.array([0.5, 1, 2, 3, 4])
    # hic.nan_bins = np.array([4])

    hic.save(outfile)

    cool_obj = hm.hiCMatrix(outfile)
    # nt.assert_equal(hic.correction_factors, cool_obj.correction_factors)
    nt.assert_equal(hic.matrix.data, cool_obj.matrix.data)
    nt.assert_equal(hic.matrix.indices, cool_obj.matrix.indices)
    nt.assert_equal(hic.matrix.indptr, cool_obj.matrix.indptr)
    nt.assert_equal(hic.nan_bins, cool_obj.nan_bins)

    nt.assert_equal(hic.cut_intervals, cool_obj.cut_intervals)
    unlink(outfile)


def test_save_load_h5():
    outfile = '/tmp/matrix.h5'
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]
    hic = hm.hiCMatrix()
    hic.nan_bins = []
    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    # make matrix symmetric
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.fillLowerTriangle()
    # hic.correction_factors = np.array([0.5, 1, 2, 3, 4])
    # hic.nan_bins = np.array([4])

    hic.save(outfile)

    h5_obj = hm.hiCMatrix(outfile)
    # nt.assert_equal(hic.correction_factors, h5_obj.correction_factors)
    nt.assert_equal(hic.matrix.data, h5_obj.matrix.data)
    nt.assert_equal(hic.matrix.indices, h5_obj.matrix.indices)
    nt.assert_equal(hic.matrix.indptr, h5_obj.matrix.indptr)
    nt.assert_equal(hic.nan_bins, h5_obj.nan_bins)

    nt.assert_equal(hic.cut_intervals, h5_obj.cut_intervals)
    unlink(outfile)


@pytest.mark.xfail
def test_save_load_other_formats_fail():
    pMatrixFile = ROOT + 'test_matrix.hicpro'
    # pBedFileHicPro = ROOT + 'test_matrix.bed'  # no parameter for this in hiCMatrix::__init__() anyway
    # hic_matrix = hm.hiCMatrix(pMatrixFile=pMatrixFile)
    # out, err = capsys.readouterr()
    # assert out == 'matrix file not given'
    pMatrixFile = ROOT + 'test_matrix.homer'
    hm.hiCMatrix(pMatrixFile=pMatrixFile)


def test_convert_to_zscore_matrix():

    # make test matrix
    m_size = 100
    mat = np.triu(np.random.randint(0, 101, (m_size, m_size)))
    # add a number of zeros
    mat[mat < 90] = 0
    # import ipdb;ipdb.set_trace()
    mu = dict([(idx, mat.diagonal(idx).mean()) for idx in range(mat.shape[0])])  # pylint: disable=R1717
    std = dict([(idx, np.std(mat.diagonal(idx)))  # pylint: disable=R1717
                for idx in range(mat.shape[0])])

    # compute z-score for test matrix
    zscore_mat = np.zeros((m_size, m_size))
    for _i in range(mat.shape[0]):
        for _j in range(mat.shape[0]):
            if _j >= _i:
                diag = _j - _i
                if std[diag] == 0:
                    zscore = np.nan
                else:
                    zscore = (mat[_i, _j] - mu[diag]) / std[diag]
                zscore_mat[_i, _j] = zscore

    # make Hi-C matrix based on test matrix
    hic = hm.hiCMatrix()
    hic.matrix = csr_matrix(mat)
    cut_intervals = [('chr', idx, idx + 10, 0) for idx in range(0, mat.shape[0] * 10, 10)]
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.convert_to_zscore_matrix()

    nt.assert_almost_equal(hic.matrix.todense(), zscore_mat)


def test_convert_to_zscore_matrix_2():

    # load test matrix
    hic = hm.hiCMatrix(ROOT + 'Li_et_al_2015.h5')
    hic.maskBins(hic.nan_bins)

    mat = hic.matrix.todense()
    max_depth = 10000
    bin_size = hic.getBinSize()
    max_depth_in_bins = int(float(max_depth) / bin_size)

    m_size = mat.shape[0]
    # compute matrix values per distance
    _, start, _, _ = list(zip(
        *hm.hiCMatrix.fit_cut_intervals(hic.cut_intervals)))
    dist_values = {}
    sys.stderr.write("Computing values per distance for each matrix entry\n")

    for _i in range(mat.shape[0]):
        for _j in range(mat.shape[0]):
            if _j >= _i:
                # dist is translated to bins
                dist = int(float(start[_j] - start[_i]) / bin_size)
                if dist <= max_depth_in_bins:
                    if dist not in dist_values:
                        dist_values[dist] = []
                    dist_values[dist].append(mat[_i, _j])

    mu = {}
    std = {}
    for dist, values in iteritems(dist_values):
        mu[dist] = np.mean(values)
        std[dist] = np.std(values)

    # compute z-score for test matrix
    sys.stderr.write("Computing zscore for each matrix entry\n")
    zscore_mat = np.full((m_size, m_size), np.nan)
    for _i in range(mat.shape[0]):
        for _j in range(mat.shape[0]):
            if _j >= _i:
                dist = int(float(start[_j] - start[_i]) / bin_size)
                if dist <= max_depth_in_bins:
                    zscore = (mat[_i, _j] - mu[dist]) / std[dist]
                    zscore_mat[_i, _j] = zscore

    # compare with zscore from class
    hic.convert_to_zscore_matrix(maxdepth=max_depth)

    # from numpy.testing import assert_almost_equal
    # only the main diagonal is check. Other diagonals show minimal differences
    nt.assert_almost_equal(hic.matrix.todense().diagonal(
        0).A1, zscore_mat.diagonal(0))


def test_dist_list_to_dict():
    hic = hm.hiCMatrix()

    data = np.array([1, 8, 5, 3, 0, 4, 15, 5, 1, 0, 0, 2, 0, 1, 0])
    dist_list = np.array(
        [0, 10, 20, 30, -1, 0, 10, 20, -1, 0, 10, -1, 0, -1, 0])

    distance = hic.dist_list_to_dict(data, dist_list)

    nt.assert_equal(distance[-1], [0, 1, 2, 1])
    nt.assert_equal(distance[0], [1, 4, 0, 0, 0])
    nt.assert_equal(distance[10], [8, 15, 0])
    nt.assert_equal(distance[20], [5, 5])
    nt.assert_equal(distance[30], [3])

    data = np.array([0, 100, 200, 0, 100, 200, 0, 100, 0])
    dist_list = np.array([0, 100, 200, 0, 100, 200, 0, 100, 0])

    distance = hic.dist_list_to_dict(data, dist_list)

    nt.assert_equal(distance[0], [0, 0, 0, 0])
    nt.assert_equal(distance[100], [100, 100, 100])
    nt.assert_equal(distance[200], [200, 200])


def test_keepOnlyTheseChr():
    chromosome_list = ['chrX', 'chr2RHet']

    hic = hm.hiCMatrix(ROOT + 'small_test_matrix.h5')

    hic.keepOnlyTheseChr(chromosome_list)

    nt.assert_equal(hic.getChrNames().sort(), chromosome_list.sort())


def test_save():
    """
    Test will not cover testing of following formats due to unsupported file_formats (see __init__ of class hiCMatrix):

    * ren
    * lieberman
    * GInteractions

    see also single test for these formats (marked as xfail)
    """

    outfile_cool = NamedTemporaryFile(suffix='.cool', delete=False)  # pylint: disable=R1732
    outfile_cool.close()

    outfile_h5 = NamedTemporaryFile(suffix='.h5', delete=False)  # pylint: disable=R1732
    outfile_h5.close()

    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.fillLowerTriangle()

    # test .h5
    hic.save(outfile_h5.name)
    h5_test = hm.hiCMatrix(outfile_h5.name)

    # test cool
    hic.matrixFileHandler = None
    hic.save(outfile_cool.name)
    cool_test = hm.hiCMatrix(outfile_cool.name)

    nt.assert_equal(hic.getMatrix(), h5_test.getMatrix())
    nt.assert_equal(hic.getMatrix(), cool_test.getMatrix())


def test_diagflat():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.fillLowerTriangle()

    hic.diagflat(value=1000)
    nt.assert_equal(
        np.array([1000 for x in range(matrix.shape[0])]), hic.matrix.diagonal())

    hic.diagflat()
    nt.assert_equal(
        np.array([np.nan for x in range(5)]), hic.matrix.diagonal())


def test_filterOutInterChrCounts():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.fillLowerTriangle()
    hic.filterOutInterChrCounts()

    filtered_matrix = np.array([[1, 8, 5, 0, 0],
                                [8, 4, 15, 0, 0],
                                [5, 15, 0, 0, 0],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0]])

    nt.assert_equal(hic.getMatrix(), filtered_matrix)

    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]
    hic = hm.hiCMatrix()
    hic.nan_bins = []
    matrix = np.array([[0, 10, 5, 3, 0],
                       [0, 0, 15, 5, 1],
                       [0, 0, 0, 7, 3],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    # make the matrix symmetric:
    hic.matrix = csr_matrix(matrix + matrix.T)
    hic.setMatrix(csr_matrix(matrix + matrix.T, dtype=np.int32), cut_intervals)

    filtered = hic.filterOutInterChrCounts().todense()
    test_matrix = np.array([[0, 10, 5, 0, 0],
                            [10, 0, 15, 0, 0],
                            [5, 15, 0, 0, 0],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 1, 0]], dtype='i4')

    nt.assert_equal(filtered, test_matrix)


def test_setMatrixValues_success():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    new_matrix = np.array([[10, 80, 50, 30, 0],
                           [0, 40, 150, 50, 10],
                           [0, 0, 0, 0, 20],
                           [0, 0, 0, 0, 10],
                           [0, 0, 0, 0, 0]])

    hic.setMatrixValues(new_matrix)

    nt.assert_equal(hic.getMatrix(), new_matrix)


def test_setMatrixValues_fail():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1)]

    new_matrix = np.array([[10, 80, 50, 30],
                           [0, 40, 150, 50],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
    with pytest.raises(AssertionError):
        hic.setMatrixValues(new_matrix)


def test_setCorrectionFactors_success():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    assert hic.correction_factors is None

    hic.setCorrectionFactors([5, 5, 5, 5, 5])

    nt.assert_equal(hic.correction_factors, [5, 5, 5, 5, 5])


def test_setCorrectionFactors_fail():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    assert hic.correction_factors is None
    with pytest.raises(AssertionError):
        hic.setCorrectionFactors([5, 5, 5, 5])


def test_reorderChromosomes():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    new_chr_order = ['b', 'a']
    hic.reorderChromosomes(new_chr_order)

    nt.assert_equal(hic.chrBinBoundaries, OrderedDict(
        [('b', (0, 2)), ('a', (2, 5))]))

    old_chr_order = ['a', 'b']
    hic.reorderChromosomes(old_chr_order)

    nt.assert_equal(hic.chrBinBoundaries, OrderedDict(
        [('a', (0, 3)), ('b', (3, 5))]))


def test_reorderChromosomes_fail():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    # name 'c' not in chromosome names, thus fail
    false_chr_order = ['a', 'b', 'c']
    with pytest.raises(Exception) as context:
        hic.reorderChromosomes(false_chr_order)
    assert "Chromosome name 'c' not found." in str(context.value)


def test_reorderBins():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    new_order = [0, 1, 3, 2, 4]
    new_matrix = np.array([[1, 8, 3, 5, 0],
                           [0, 4, 5, 15, 1],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 2],
                           [0, 0, 0, 0, 0]])

    hic.reorderBins(new_order)

    nt.assert_equal(hic.getMatrix(), new_matrix)

    hic.reorderBins(new_order)

    nt.assert_equal(hic.getMatrix(), matrix)

    # order smaller than original matrix should delete unused ids
    small_order = [2, 3]
    small_matrix = np.array([[0, 0],
                             [0, 0]])

    hic.reorderBins(small_order)

    nt.assert_equal(hic.getMatrix(), small_matrix)
    nt.assert_equal(hic.matrix.shape, small_matrix.shape)
    nt.assert_equal(hic.chrBinBoundaries, OrderedDict(
        [('a', (0, 1)), ('b', (1, 2))]))
    nt.assert_equal(hic.cut_intervals, [('a', 20, 30, 1), ('b', 30, 40, 1)])
    nt.assert_equal(hic.nan_bins, [])


def test_maskBins():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)
    nt.assert_equal(hic.orig_bin_ids, [])

    new_matrix = np.array([[0, 0, 2],
                           [0, 0, 1],
                           [0, 0, 0]])

    masking_ids = [0, 1]
    hic.maskBins(masking_ids)

    nt.assert_equal(hic.getMatrix(), new_matrix)
    nt.assert_equal(sorted(hic.orig_cut_intervals), sorted([('a', 0, 10, 1), ('a', 10, 20, 1),
                                                            ('a', 20, 30,
                                                             1), ('b', 30, 40, 1),
                                                            ('b', 40, 50, 1)]))
    nt.assert_equal(sorted(hic.cut_intervals), sorted([('a', 20, 30, 1), ('b', 30, 40, 1),
                                                       ('b', 40, 50, 1)]))
    nt.assert_equal(hic.chrBinBoundaries, OrderedDict(
        [('a', (0, 1)), ('b', (1, 3))]))
    nt.assert_equal(sorted(hic.orig_bin_ids), sorted([0, 1, 2, 3, 4]))

    # direct return if masking_ids is None or has len() == 0, thus no changes to matrix
    masking_ids = None
    hic.maskBins(masking_ids)

    nt.assert_equal(hic.getMatrix(), new_matrix)
    nt.assert_equal(sorted(hic.orig_cut_intervals), sorted([('a', 0, 10, 1), ('a', 10, 20, 1),
                                                            ('a', 20, 30,
                                                             1), ('b', 30, 40, 1),
                                                            ('b', 40, 50, 1)]))
    nt.assert_equal(sorted(hic.cut_intervals), sorted([('a', 20, 30, 1), ('b', 30, 40, 1),
                                                       ('b', 40, 50, 1)]))
    nt.assert_equal(hic.chrBinBoundaries, OrderedDict(
        [('a', (0, 1)), ('b', (1, 3))]))

    masking_ids = []

    hic.maskBins(masking_ids)

    nt.assert_equal(hic.getMatrix(), new_matrix)
    nt.assert_equal(sorted(hic.orig_cut_intervals), sorted([('a', 0, 10, 1), ('a', 10, 20, 1),
                                                            ('a', 20, 30,
                                                             1), ('b', 30, 40, 1),
                                                            ('b', 40, 50, 1)]))
    nt.assert_equal(sorted(hic.cut_intervals), sorted([('a', 20, 30, 1), ('b', 30, 40, 1),
                                                       ('b', 40, 50, 1)]))
    nt.assert_equal(hic.chrBinBoundaries, OrderedDict(
        [('a', (0, 1)), ('b', (1, 3))]))

    nt.assert_equal(sorted(hic.orig_bin_ids), sorted([0, 1, 2, 3, 4]))


def test_update_matrix():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    new_cut_intervals = [('c', 0, 10, 1), ('d', 10, 20, 1), ('d', 20, 30, 1)]

    new_matrix = np.array([[3, 6, 4],
                           [np.nan, 0, 2],
                           [1, 0, 0]])
    try:
        hic.update_matrix(new_matrix, new_cut_intervals)
    except AttributeError:
        pass
    # if matrix.shape[0] not equal to length of cut_intervals assertionError is raised
    short_cut_intervals = [('c', 0, 10, 1), ('d', 10, 20, 1)]

    with pytest.raises(AssertionError):
        hic.update_matrix(new_matrix, short_cut_intervals)

    # if matrix contains masked bins exception is raised
    masking_ids = [0, 1]
    hic.maskBins(masking_ids)

    with pytest.raises(Exception):
        hic.update_matrix(new_matrix, new_cut_intervals)


def test_restoreMaskedBins():
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)
    nt.assert_equal(hic.orig_bin_ids, [])

    # function should directly return if there are no masked_bins
    hic.restoreMaskedBins()

    nt.assert_equal(hic.getMatrix(), matrix)
    nt.assert_equal(hic.orig_bin_ids, [])

    # test general use
    # first get some masked bins
    masking_ids = [0, 1]
    hic.maskBins(masking_ids)

    new_matrix = np.array([[0, 0, 2],
                           [0, 0, 1],
                           [0, 0, 0]])

    nt.assert_equal(hic.getMatrix(), new_matrix)
    nt.assert_equal(sorted(hic.orig_bin_ids), sorted([0, 1, 2, 3, 4]))

    # and now restore masked bins
    hic.restoreMaskedBins()

    result_matrix = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, 0, 0, 2],
                              [np.nan, np.nan, 0, 0, 1],
                              [np.nan, np.nan, 0, 0, 0]])

    nt.assert_equal(hic.getMatrix(), result_matrix)
    nt.assert_equal(hic.orig_bin_ids, [])

    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]
    hic = hm.hiCMatrix()
    hic.nan_bins = []
    matrix = np.array([[0, 10, 5, 3, 0],
                       [0, 0, 15, 5, 1],
                       [0, 0, 0, 7, 3],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]], dtype=np.int32)

    # make the matrix symmetric:
    hic.matrix = csr_matrix(matrix + matrix.T)
    hic.setMatrix(csr_matrix(matrix + matrix.T), cut_intervals)

    # Add masked bins masked bins
    hic.maskBins([3])

    matrix = hic.matrix.todense()
    test_matrix = np.array([[0, 10, 5, 0],
                            [10, 0, 15, 1],
                            [5, 15, 0, 3],
                            [0, 1, 3, 0]], dtype=np.int32)

    nt.assert_equal(matrix, test_matrix)

    cut_int = hic.cut_intervals
    test_cut_int = [('a', 0, 10, 1), ('a', 10, 20, 1), ('a', 20, 30, 1), ('b', 40, 50, 1)]

    nt.assert_equal(cut_int, test_cut_int)

    hic.restoreMaskedBins()

    dense = hic.matrix.todense()
    test_dense = np.array([[0., 10., 5., 0., 0.],
                           [10., 0., 15., 0., 1.],
                           [5., 15., 0., 0., 3.],
                           [0., 0., 0., 0., 0.],
                           [0., 1., 3., 0., 0.]])

    nt.assert_equal(dense, test_dense)

    cut_int = hic.cut_intervals
    test_cut_int = [('a', 0, 10, 1), ('a', 10, 20, 1), ('a', 20, 30, 1),
                    ('a', 30, 40, 1), ('b', 40, 50, 1)]

    nt.assert_equal(cut_int, test_cut_int)


def test_reorderMatrix():
    orig = (1, 3)
    dest = 2

    # get matrix
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    # reorder matrix
    hic.reorderMatrix(orig, dest)

    new_matrix = np.array([[1, 3, 8, 5, 0],
                           [0, 0, 0, 0, 1],
                           [0, 5, 4, 15, 1],
                           [0, 0, 0, 0, 2],
                           [0, 0, 0, 0, 0]])

    new_cut_intervals = [('a', 0, 10, 1), ('b', 30, 40, 1),
                         ('a', 10, 20, 1), ('a', 20, 30, 1), ('b', 40, 50, 1)]

    # check if it is equal
    nt.assert_equal(hic.getMatrix(), new_matrix)
    nt.assert_equal(hic.matrix.shape, new_matrix.shape)
    nt.assert_equal(hic.cut_intervals, new_cut_intervals)


def test_truncTrans():
    # get matrix
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[-1, 8, 5, 3, 0],
                       [np.nan, 4, 15, 5, 100],
                       [0, 0, 0, 0, 2000],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    # define expected outcome
    new_matrix = np.array([[-1., 8., 5., 3., 0.],
                           [np.nan, 4., 15., 5., 1.e+2],
                           [0., 0., 0., 0., 2.e+3],
                           [0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0.]])

    # truncTrans of matrix
    hic.truncTrans()

    # test against expected outcome
    nt.assert_equal(hic.getMatrix(), new_matrix)

    # reset matrix
    matrix = np.array([[-1, 8, 5, 3, 0],
                       [np.nan, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    # method should directly return if nothing to do, matrix stays the same
    hic.truncTrans()
    nt.assert_equal(hic.getMatrix(), matrix)


def test_printchrtoremove(capsys):
    # get matrix
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    # first test exception message for no self.prev_to_remove
    to_remove = [0, 1]

    with pytest.raises(Exception):
        hic.printchrtoremove(to_remove)

        captured = capsys.readouterr()
        assert captured.out == "No self.prev_to_remove defined, defining it now."

        nt.assert_equal(hic.prev_to_remove, np.array(to_remove))

    nt.assert_equal(hic.orig_bin_ids, [])

    # also test with masked_bins
    hic.maskBins(to_remove)

    assert len(hic.orig_bin_ids) > 0

    hic.printchrtoremove(to_remove)

    nt.assert_equal(hic.prev_to_remove, np.array(to_remove))


def test_get_chromosome_sizes_real():
    # get matrix
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    # define expected outcome
    expected_sizes = OrderedDict([('a', 31), ('b', 21)])

    chrom_sizes = hic.get_chromosome_sizes_real()

    nt.assert_equal(chrom_sizes, expected_sizes)

    # define new intervals and test again
    new_cut_intervals = [('a', 0, 10, 1), ('b', 10, 20, 1),
                         ('b', 20, 30, 1), ('c', 30, 40, 1), ('c', 40, 90, 1)]

    expected_sizes = OrderedDict([('a', 11), ('b', 21), ('c', 61)])

    hic.setMatrix(hic.matrix, new_cut_intervals)

    chrom_sizes = hic.get_chromosome_sizes_real()

    nt.assert_equal(chrom_sizes, expected_sizes)


def test_get_chromosome_sizes():
    # get matrix
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    # define expected outcome
    expected_sizes = OrderedDict([('a', 30), ('b', 50)])

    chrom_sizes = hic.get_chromosome_sizes()

    nt.assert_equal(chrom_sizes, expected_sizes)

    # define new intervals and test again
    new_cut_intervals = [('a', 0, 10, 1), ('b', 10, 20, 1),
                         ('b', 20, 30, 1), ('c', 30, 40, 1), ('c', 40, 90, 1)]

    expected_sizes = OrderedDict([('a', 10), ('b', 30), ('c', 90)])

    hic.setMatrix(hic.matrix, new_cut_intervals)

    chrom_sizes = hic.get_chromosome_sizes()

    nt.assert_equal(chrom_sizes, expected_sizes)


def test_intervalListToIntervalTree(capsys):
    # get matrix
    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    nt.assert_equal(hic.getMatrix(), matrix)

    # empty list should raise AssertionError
    interval_list = []
    with pytest.raises(AssertionError):
        hic.intervalListToIntervalTree(interval_list)

        captured = capsys.readouterr()
        assert captured.out == "Interval list is empty"

    # test with correct interval_list
    interval_list = [('a', 0, 10, 1), ('a', 10, 20, 1), ('b', 20, 30, 1), ('b', 30, 50, 1),
                     ('b', 50, 100, 1), ('c', 100, 200, 1), ('c', 200, 210, 1),
                     ('d', 210, 220, 1), ('e', 220, 250)]

    tree, boundaries = hic.intervalListToIntervalTree(interval_list)

    # test tree
    nt.assert_equal(tree['a'], IntervalTree([Interval(0, 10, 0), Interval(10, 20, 1)]))
    nt.assert_equal(tree['b'], IntervalTree([Interval(20, 30, 2), Interval(30, 50, 3),
                                             Interval(50, 100, 4)]))
    nt.assert_equal(tree['c'], IntervalTree([Interval(100, 200, 5), Interval(200, 210, 6)]))
    nt.assert_equal(tree['d'], IntervalTree([Interval(210, 220, 7)]))
    nt.assert_equal(tree['e'], IntervalTree([Interval(220, 250, 8)]))

    # test boundaries
    nt.assert_equal(boundaries, OrderedDict([('a', (0, 2)), ('b', (2, 5)), ('c', (5, 7)),
                                             ('d', (7, 8)), ('e', (8, 9))]))


def test_fillLowerTriangle():
    A = csr_matrix(np.array([[12, 5, 3, 2, 0], [0, 11, 4, 1, 1], [0, 0, 9, 6, 0],
                             [0, 0, 0, 10, 0], [0, 0, 0, 0, 0]]), dtype=np.int32)
    hic = hm.hiCMatrix()
    hic.matrix = A
    hic.fillLowerTriangle()
    B = hic.matrix
    test_matrix = np.array([[12, 5, 3, 2, 0],
                            [5, 11, 4, 1, 1],
                            [3, 4, 9, 6, 0],
                            [2, 1, 6, 10, 0],
                            [0, 1, 0, 0, 0]], dtype='i4')

    nt.assert_equal(B.todense(), test_matrix)


def test_getDistList():
    row, col = np.triu_indices(5)
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]
    dist_list, chrom_list = hm.hiCMatrix.getDistList(row, col, cut_intervals)

    matrix = coo_matrix((dist_list, (row, col)), shape=(5, 5), dtype=np.int32).todense()
    test_matrix = np.array([[0, 10, 20, 30, -1],
                            [0, 0, 10, 20, -1],
                            [0, 0, 0, 10, -1],
                            [0, 0, 0, 0, -1],
                            [0, 0, 0, 0, 0]], dtype='i4')
    nt.assert_equal(matrix, test_matrix)

    chrom_list = chrom_list.tolist()
    test_chrom_list = ['a', 'a', 'a', 'a', '', 'a', 'a', 'a', '', 'a', 'a', '', 'a',
                       '', 'b']

    nt.assert_equal(chrom_list, test_chrom_list)


def test_convert_to_obs_exp_matrix():
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('a', 30, 40, 1), ('b', 40, 50, 1)]
    hic = hm.hiCMatrix()
    hic.nan_bins = []
    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 7, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    obs_exp_matrix = hic.convert_to_obs_exp_matrix().todense()
    test_matrix = np.array([[1., 0.8, 1., 1., 0.],
                            [0., 4., 1.5, 1., 1.],
                            [0., 0., 0., 0.7, 2.],
                            [0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0.]])

    nt.assert_almost_equal(obs_exp_matrix, test_matrix)

    hic.matrix = csr_matrix(matrix)
    obs_exp_matrix = hic.convert_to_obs_exp_matrix(maxdepth=20).todense()
    test_matrix = np.array([[1., 0.8, 1., 0., 0.],
                            [0., 4., 1.5, 1., 0.],
                            [0., 0., 0., 0.7, np.nan],
                            [0., 0., 0., 0., np.nan],
                            [0., 0., 0., 0., 0.]])

    nt.assert_almost_equal(obs_exp_matrix, test_matrix)

    hic.matrix = csr_matrix(matrix)

    obs_exp_matrix = hic.convert_to_obs_exp_matrix(zscore=True).todense()
    test_matrix = np.array([[0., -0.56195149, np.nan, np.nan, -1.41421356],
                            [0., 1.93649167, 1.40487872, np.nan, 0.],
                            [0., 0., -0.64549722, -0.84292723, 1.41421356],
                            [0., 0., 0., -0.64549722, 0.],
                            [0., 0., 0., 0., -0.64549722]])

    nt.assert_almost_equal(obs_exp_matrix, test_matrix)


def test_maskChromosomes():

    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)
    hic.maskChromosomes(['a'])


@pytest.mark.xfail
def test_maskChromosomes_fail():

    hic = hm.hiCMatrix()
    cut_intervals = [('a', 0, 10, 1), ('a', 10, 20, 1),
                     ('a', 20, 30, 1), ('b', 30, 40, 1), ('b', 40, 50, 1)]

    hic.nan_bins = []

    matrix = np.array([[1, 8, 5, 3, 0],
                       [0, 4, 15, 5, 1],
                       [0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])

    hic.matrix = csr_matrix(matrix)
    hic.setMatrix(hic.matrix, cut_intervals)

    hic.maskChromosomes(['c'])

    print(hic.matrix)


def test_create_from_cool():
    hic_ma = hm.hiCMatrix(ROOT + 'one_interaction_4chr.cool')
    nt.assert_equal(sorted(hic_ma.matrix.indices), [0, 3])
    nt.assert_equal(sorted(hic_ma.matrix.data), [1, 1])
    nt.assert_equal(sorted(hic_ma.nan_bins)[:5], [1, 2, 4, 5, 6])
    hic_ma = hm.hiCMatrix(ROOT + 'one_interaction_diag_4chr.cool')
    nt.assert_equal(sorted(hic_ma.matrix.indices), [0])
    nt.assert_equal(sorted(hic_ma.matrix.data), [1])
    nt.assert_equal(sorted(hic_ma.nan_bins)[:5], [1, 2, 3, 4, 5])
    hic_ma.maskBins(hic_ma.nan_bins)
    assert hic_ma.matrix.shape == (1, 1)
    assert hic_ma.getBinSize() == 50000


def test_load_cool_matrix_only():
    hic_cool = hm.hiCMatrix(ROOT + 'Li_et_al_2015.cool', pUpperTriangleOnly=True)

    hic_cool_matrix_only = hm.hiCMatrix(ROOT + 'Li_et_al_2015.cool', pUpperTriangleOnly=True, pLoadMatrixOnly=True)
    instances = hic_cool_matrix_only.matrix[0]
    features = hic_cool_matrix_only.matrix[1]
    data = hic_cool_matrix_only.matrix[2]

    instances_cool, features_cool = hic_cool.matrix.nonzero()
    nt.assert_equal(hic_cool.matrix.data, data)
    nt.assert_equal(instances_cool, instances)
    nt.assert_equal(features_cool, features)
