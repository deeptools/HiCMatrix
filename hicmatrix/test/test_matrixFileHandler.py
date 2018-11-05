from hicmatrix.lib import MatrixFileHandler
import numpy.testing as nt
import os
import pytest
from scipy.sparse.csr import csr_matrix
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")
outfile = '/tmp/matrix'


def test_load_homer(capsys):
    # create matrixFileHandler instance with filetype 'homer'
    pMatrixFile = ROOT + 'test_matrix.homer'
    fh = MatrixFileHandler(pFileType='homer', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # create test matrix

    test_matrix = np.array([[1.0, 0.1896, 0.2163, 0.08288, 0.1431, 0.2569, 0.1315,
                             0.1488, -0.0312, 0.143, 0.06091, 0.03546, 0.1168]])

    nt.assert_almost_equal(matrix[0].todense(), test_matrix)

    test_cut_intervals = [('3R', 1000000, 1020000, 1), ('3R', 1020000, 1040000, 1), ('3R', 1040000, 1060000, 1), ('3R', 1060000, 1080000, 1), ('3R', 1080000, 1100000, 1), ('3R', 1100000, 1120000, 1), ('3R', 1120000, 1140000, 1), ('3R', 1140000, 1160000, 1), ('3R', 1160000, 1180000, 1), ('3R', 1180000, 1200000, 1), ('3R', 1200000, 1220000, 1), ('3R', 1220000, 1240000, 1), ('3R', 1240000, 1260000, 1)]  # noqa E501
    nt.assert_equal(cut_intervals, test_cut_intervals)

    assert nan_bins is None
    assert distance_counts is None
    assert correction_factors is None


def test_save_homer():
    homer_outfile = outfile + '.homer'

    # create matrixFileHandler instance with filetype 'homer'
    pMatrixFile = ROOT + 'test_matrix.homer'
    fh = MatrixFileHandler(pFileType='homer', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)  # noqa E501
    # and save it.
    fh.save(pName=homer_outfile, pSymmetric=False, pApplyCorrection=False)  # not implemented
    os.unlink(homer_outfile)


def test_load_h5(capsys):
    # create matrixFileHandler instance with filetype 'h5'
    pMatrixFile = ROOT + 'Li_et_al_2015.h5'
    fh = MatrixFileHandler(pFileType='h5', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    test_matrix = np.array([[0. for i in range(11104)]])
    nt.assert_almost_equal(matrix[0].todense(), test_matrix)

    nt.assert_equal(cut_intervals[0], ('X', 0, 2200, 0.0))
    nt.assert_equal(cut_intervals[1], ('X', 2200, 4702, 0.0))
    nt.assert_equal(cut_intervals[2], ('X', 4702, 7060, 0.0))
    nt.assert_equal(cut_intervals[3], ('X', 7060, 8811, 0.4))

    test_nan_bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 30, 31, 32, 51, 52, 53, 54, 81, 82, 83, 84, 94])  # noqa E501
    nt.assert_equal(nan_bins[0:20], test_nan_bins)

    assert distance_counts is None

    test_correction_factors = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.90720049, 1.25516028])  # noqa E501
    nt.assert_almost_equal(correction_factors[0:10], test_correction_factors)


def test_save_h5():
    h5_outfile = outfile + '.h5'

    # create matrixFileHandler instance with filetype 'h5'
    pMatrixFile = ROOT + 'Li_et_al_2015.h5'
    fh = MatrixFileHandler(pFileType='h5', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)  # noqa E501
    # and save it.
    fh.save(h5_outfile, True, None)

    os.unlink(h5_outfile)


def test_load_hicpro(capsys):
    # create matrixFileHandler instance with filetype 'hicpro'
    pMatrixFile = ROOT + 'test_matrix.hicpro'
    pBedFileHicPro = ROOT + 'test_matrix.bed'
    fh = MatrixFileHandler(pFileType='hicpro', pMatrixFile=pMatrixFile, pBedFileHicPro=pBedFileHicPro)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # create test matrix
    test_list = [0. for i in range(3113)]
    test_list.insert(0, 41.345793)
    test_list[827] = 5.42079
    test_list[1263] = 5.122642

    test_matrix = np.matrix([test_list])

    # and check for shape and values
    assert matrix[0].todense().shape == test_matrix.shape
    nt.assert_almost_equal(matrix[0].todense(), test_matrix)

    test_cut_intervals = np.array([('chr1', 0, 1000000, 1), ('chr1', 1000000, 2000000, 2), ('chr1', 2000000, 3000000, 3),
                                   ('chr1', 3000000, 4000000, 4), ('chr1', 4000000, 5000000, 5), ('chr1', 5000000, 6000000, 6),
                                   ('chr1', 6000000, 7000000, 7), ('chr1', 7000000, 8000000, 8), ('chr1', 8000000, 9000000, 9),
                                   ('chr1', 9000000, 10000000, 10), ('chr1', 10000000, 11000000, 11), ('chr1', 11000000, 12000000, 12),
                                   ('chr1', 12000000, 13000000, 13), ('chr1', 13000000, 14000000, 14), ('chr1', 14000000, 15000000, 15),
                                   ('chr1', 15000000, 16000000, 16), ('chr1', 16000000, 17000000, 17), ('chr1', 17000000, 18000000, 18),
                                   ('chr1', 18000000, 19000000, 19), ('chr1', 19000000, 20000000, 20)])
    nt.assert_equal(cut_intervals[0:20], test_cut_intervals)

    assert nan_bins is None
    assert correction_factors is None
    assert distance_counts is None


@pytest.mark.xfail
def test_save_hicpro():
    hicpro_outfile = outfile + '.hicpro'

    # create matrixFileHandler instance with filetype 'hicpro'
    pMatrixFile = ROOT + 'test_matrix.hicpro'
    pBedFileHicPro = ROOT + 'test_matrix.bed'
    fh = MatrixFileHandler(pFileType='hicpro', pMatrixFile=pMatrixFile, pBedFileHicPro=pBedFileHicPro)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=hicpro_outfile, pSymmetric=False, pApplyCorrection=False)  # not implemented
    os.unlink(hicpro_outfile)


def test_load_cool(capsys):
    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # test matrix
    test_matrix = np.matrix([[0. for i in range(11104)]])
    nt.assert_almost_equal(matrix[0].todense(), test_matrix)

    test_cut_intervals = [('X', 0, 2200, 1.0), ('X', 2200, 4702, 1.0), ('X', 4702, 7060, 1.0),
                          ('X', 7060, 8811, 1.0), ('X', 8811, 11048, 1.0), ('X', 11048, 14329, 1.0),
                          ('X', 14329, 16847, 1.0), ('X', 16847, 19537, 1.0), ('X', 19537, 20701, 1.0),
                          ('X', 20701, 22321, 1.0), ('X', 22321, 24083, 1.0), ('X', 24083, 25983, 1.0),
                          ('X', 25983, 27619, 1.0), ('X', 27619, 29733, 1.0), ('X', 29733, 30973, 1.0),
                          ('X', 30973, 32214, 1.0), ('X', 32214, 34179, 1.0), ('X', 34179, 35987, 1.0),
                          ('X', 35987, 37598, 1.0), ('X', 37598, 39009, 1.0)]
    for index, tup in enumerate(cut_intervals[0:20]):
        for ind, element in enumerate(tup):
            assert element == test_cut_intervals[index][ind]

    test_nan_bins = [0, 1, 2, 3, 4, 5, 6, 7, 30, 31]
    nt.assert_almost_equal(nan_bins[0:10], test_nan_bins)

    test_correction_factors = [1., 1., 1., 1., 1., 1., 1., 1., 0.90720049, 1.25516028]
    nt.assert_almost_equal(correction_factors[0:10], test_correction_factors)

    assert distance_counts is None


def test_save_cool():
    cool_outfile = outfile + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=True, pApplyCorrection=True)

    os.unlink(cool_outfile)



def test_save_cool_non_symmetric_apply_correction_true():
    cool_outfile = outfile + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=True)

    os.unlink(cool_outfile)


def test_save_cool_non_symmetric_apply_correction_false():
    cool_outfile = outfile + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=False)

    os.unlink(cool_outfile)


def test_save_cool_enforce_integer():
    cool_outfile = outfile + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pEnforceInteger=True)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=False)

    os.unlink(cool_outfile)


def test_save_cool_apply_division_none_correction():
    cool_outfile = outfile + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pCorrectionOperator='/')
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # print(correction_factors)
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, None, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=False)

    os.unlink(cool_outfile)

def test_save_cool_apply_division():
    cool_outfile = outfile + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pCorrectionOperator='/')
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # print(correction_factors)
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=False)

    os.unlink(cool_outfile)
    
def test_load_ginteractions():
    pass


def test_save_ginteractions():
    pass
