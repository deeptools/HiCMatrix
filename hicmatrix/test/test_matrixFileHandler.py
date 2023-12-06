import logging
import os
from tempfile import NamedTemporaryFile

import cooler
import numpy as np
import numpy.testing as nt
import pytest

from hicmatrix.lib import MatrixFileHandler

log = logging.getLogger(__name__)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")
outfile_basename = '/tmp/matrix'


def test_load_homer():
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


def test_load_homer_gzip():
    # create matrixFileHandler instance with filetype 'homer'
    pMatrixFile = ROOT + 'test_matrix.homer.gz'
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
    homer_outfile = outfile_basename + '.homer'

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


def test_load_h5():
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

    test_correction_factors = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.90720049, 1.25516028])  # noqa E501
    nt.assert_almost_equal(correction_factors[0:10], test_correction_factors)


def test_save_h5():
    h5_outfile = outfile_basename + '.h5'

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


def test_load_hicpro():
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

    test_matrix = np.array([test_list])

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
    hicpro_outfile = outfile_basename + '.hicpro'

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


def test_load_cool():
    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # test matrix
    test_matrix = np.array([[0. for i in range(11104)]])
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

    test_correction_factors = [0., 0., 0., 0., 0., 0., 0., 0., 1.1022922, 0.796711]
    nt.assert_almost_equal(correction_factors[0:10], test_correction_factors)

    assert distance_counts is None


def test_load_cool2():
    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'one_interaction_4chr.cool'
    # The interaction is:
    # chr1	10000	chr1	200000
    bin_size = 50000
    # So there should be a 1 between the bin 0 and the bin 3
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # test data
    nt.assert_almost_equal(matrix.data, np.array([1]))

    # test matrix
    test_matrix = np.array([[0 for i in range(9167)]])
    nt.assert_almost_equal(matrix[3].todense(), test_matrix)
    test_matrix[0][3] = 1
    nt.assert_almost_equal(matrix[0].todense(), test_matrix)

    test_cut_intervals = sum([[('chr1', i * bin_size, (i + 1) * bin_size, 1.0) for i in range(3909)],
                              [('chr1', 195450000, 195471971, 1.0)],
                              [('chrX', i * bin_size, (i + 1) * bin_size, 1.0) for i in range(3420)],
                              [('chrX', 171000000, 171031299, 1.0)],
                              [('chrY', i * bin_size, (i + 1) * bin_size, 1.0) for i in range(1834)],
                              [('chrY', 91700000, 91744698, 1.0)],
                              [('chrM', 0, 16299, 1.0)]], [])

    for index, tup in enumerate(cut_intervals):
        for ind, element in enumerate(tup):
            assert element == test_cut_intervals[index][ind]

    test_nan_bins = [1, 2, 4, 5]
    nt.assert_almost_equal(nan_bins[:4], test_nan_bins)

    assert distance_counts is None
    assert correction_factors is None


def test_save_cool():
    cool_outfile = outfile_basename + '.cool'

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

    fh_test = MatrixFileHandler(pFileType='cool', pMatrixFile=cool_outfile)
    assert fh_test is not None
    matrix_test, cut_intervals_test, nan_bins_test, distance_counts_test, correction_factors_test = fh_test.load()

    nt.assert_equal(matrix.data, matrix_test.data)
    nt.assert_equal(cut_intervals, cut_intervals_test)
    nt.assert_equal(nan_bins, nan_bins_test)
    nt.assert_equal(distance_counts, distance_counts_test)
    nt.assert_equal(correction_factors, correction_factors_test)

    os.unlink(cool_outfile)


def test_load_distance_cool():
    cool_outfile = outfile_basename + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool051.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pChrnameList=['1'], pDistance=2500000)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(pName=cool_outfile, pSymmetric=True, pApplyCorrection=True)

    fh_test = MatrixFileHandler(pFileType='cool', pMatrixFile=cool_outfile)
    assert fh_test is not None
    matrix_test, cut_intervals_test, nan_bins_test, distance_counts_test, correction_factors_test = fh_test.load()

    # check distance load works as expected
    instances, features = matrix.nonzero()
    distances = np.absolute(instances - features)
    # log.debug('max: {}'.format(np.max(distances)))
    mask = distances > 1  # 2.5 mb res --> all with  2.5 Mb distance
    assert np.sum(mask) == 0

    fh = MatrixFileHandler(pFileType='cool', pChrnameList=['1'], pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix2, _, _, _, _ = fh.load()
    instances, features = matrix2.nonzero()
    distances = np.absolute(instances - features)
    mask = distances > 1  # 2.5 mb res --> all with  2.5 Mb distance
    assert np.sum(mask) > 0

    # check if load and save matrix are equal
    nt.assert_equal(matrix.data, matrix_test.data)
    nt.assert_equal(cut_intervals, cut_intervals_test)
    nt.assert_equal(nan_bins, nan_bins_test)
    nt.assert_equal(distance_counts, distance_counts_test)
    nt.assert_equal(correction_factors, correction_factors_test)

    os.unlink(cool_outfile)


def test_load_h5_save_cool():
    cool_outfile = outfile_basename + '.cool'

    # create matrixFileHandler instance with filetype 'h5'
    pMatrixFile = ROOT + 'Li_et_al_2015.h5'
    fh = MatrixFileHandler(pFileType='h5', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # set matrix variables
    fh_new = MatrixFileHandler(pFileType='cool')

    fh_new.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    fh_new.matrixFile.fileWasH5 = True
    # and save it.

    fh_new.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=True)

    fh_test = MatrixFileHandler(pFileType='cool', pMatrixFile=cool_outfile)
    assert fh_test is not None
    matrix_test, cut_intervals_test, nan_bins_test, distance_counts_test, correction_factors_test = fh_test.load()

    instances, features = matrix.nonzero()
    instances_factors = correction_factors[instances]
    features_factors = correction_factors[features]
    instances_factors *= features_factors

    matrix_applied_correction = matrix.data / instances_factors
    nt.assert_almost_equal(matrix_applied_correction, matrix_test.data, decimal=1)
    nt.assert_equal(len(cut_intervals), len(cut_intervals_test))
    nt.assert_equal(nan_bins, nan_bins_test)
    nt.assert_equal(distance_counts, distance_counts_test)
    correction_factors = 1 / correction_factors
    mask = np.isnan(correction_factors)
    correction_factors[mask] = 0
    mask = np.isinf(correction_factors)
    correction_factors[mask] = 0
    nt.assert_equal(correction_factors, correction_factors_test)

    # os.unlink(cool_outfile)
    os.unlink(cool_outfile)


def test_save_cool_enforce_integer():
    cool_outfile = outfile_basename + '.cool'

    # create matrixFileHandler instance with filetype 'h5'
    pMatrixFile = ROOT + 'Li_et_al_2015.h5'
    fh = MatrixFileHandler(pFileType='h5', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # set matrix variables
    fh_new = MatrixFileHandler(pFileType='cool', pEnforceInteger=True)

    fh_new.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    fh_new.matrixFile.fileWasH5 = True
    # and save it.

    fh_new.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=True)

    fh_test = MatrixFileHandler(pFileType='cool', pMatrixFile=cool_outfile, pApplyCorrectionCoolerLoad=False)
    assert fh_test is not None
    matrix_test, cut_intervals_test, nan_bins_test, distance_counts_test, _ = fh_test.load()

    # pMatrixFile = ROOT + 'Li_et_al_2015.h5'
    # fh = MatrixFileHandler(pFileType='h5', pMatrixFile=pMatrixFile)
    # assert fh is not None

    # load data
    # matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # instances, features = matrix.nonzero()
    # instances_factors = correction_factors[instances]
    # features_factors = correction_factors[features]
    # instances_factors *= features_factors

    # matrix_applied_correction = matrix.data / instances_factors
    # mask = matrix.data == 0
    matrix.data = np.rint(matrix.data)
    matrix.eliminate_zeros()
    # matrix_test.eliminate_zeros()

    nt.assert_almost_equal(matrix.data, matrix_test.data, decimal=0)
    nt.assert_equal(len(cut_intervals), len(cut_intervals_test))
    nt.assert_equal(nan_bins, nan_bins_test)
    nt.assert_equal(distance_counts, distance_counts_test)

    # os.unlink(cool_outfile)
    os.unlink(cool_outfile)


def test_load_cool_hic2cool_versions():
    pMatrixFile = ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool042.cool'
    hic2cool_042 = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pCorrectionFactorTable='KR', pCorrectionOperator='*')
    pMatrixFile = ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool051.cool'
    hic2cool_051 = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pCorrectionFactorTable='KR')

    # hic2cool_051 = MatrixFileHandler(pFileType='h5', pMatrixFile=, pCorrectionFactorTable='KR')
    # hic2cool_042 = hm.hiCMatrix(ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool042.cool')
    # hic2cool_051 = hm.hiCMatrix(ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool051.cool')

    # hic2cool_041 = hm.hiCMatrix(outfile.name)
    matrix, cut_intervals, nan_bins, distance_counts, _ = hic2cool_042.load()
    matrix_test, cut_intervals_test, nan_bins_test, distance_counts_test, _ = hic2cool_051.load()

    nt.assert_almost_equal(matrix.data, matrix_test.data, decimal=0)
    nt.assert_equal(len(cut_intervals), len(cut_intervals_test))
    nt.assert_equal(nan_bins, nan_bins_test)
    nt.assert_equal(distance_counts, distance_counts_test)


def test_save_cool_apply_division():
    cool_outfile = outfile_basename + '.cool'

    # create matrixFileHandler instance with filetype 'cool'
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pCorrectionOperator='/')
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh_new = MatrixFileHandler(pFileType='cool', pCorrectionOperator='/')

    fh_new.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)

    # and save it.

    fh_new.save(pName=cool_outfile, pSymmetric=False, pApplyCorrection=True)

    fh_test = MatrixFileHandler(pFileType='cool', pMatrixFile=cool_outfile)
    assert fh_test is not None
    matrix_test, cut_intervals_test, nan_bins_test, distance_counts_test, _ = fh_test.load()
    pMatrixFile = ROOT + 'Li_et_al_2015.cool'
    fh = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pCorrectionOperator='/')
    assert fh is not None
    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    nt.assert_almost_equal(matrix.data, matrix_test.data, decimal=1)
    nt.assert_equal(len(cut_intervals), len(cut_intervals_test))
    nt.assert_equal(nan_bins, nan_bins_test)
    nt.assert_equal(distance_counts, distance_counts_test)

    os.unlink(cool_outfile)


def test_save_scool_matrixHandlersCool():

    outfile = NamedTemporaryFile(suffix='.scool', prefix='hicmatrix_scool_test')  # pylint: disable=R1732

    pMatrixFile = ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool051.cool'

    matrixFileHandlerInput = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    matrix, cut_intervals, nan_bins, \
        distance_counts, correction_factors = matrixFileHandlerInput.load()
    matrixFileHandlerOutput1 = MatrixFileHandler(pFileType='cool', pMatrixFile='cell1', pEnforceInteger=False, pFileWasH5=False, pHic2CoolVersion=None)
    matrixFileHandlerOutput1.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)

    matrixFileHandlerOutput2 = MatrixFileHandler(pFileType='cool', pMatrixFile='cell2', pEnforceInteger=False, pFileWasH5=False, pHic2CoolVersion=None)
    matrixFileHandlerOutput2.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)

    matrixFileHandlerOutput3 = MatrixFileHandler(pFileType='cool', pMatrixFile='cell3', pEnforceInteger=False, pFileWasH5=False, pHic2CoolVersion=None)
    matrixFileHandlerOutput3.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)

    matrixFileHandler = MatrixFileHandler(pFileType='scool')
    matrixFileHandler.matrixFile.coolObjectsList = [matrixFileHandlerOutput1, matrixFileHandlerOutput2, matrixFileHandlerOutput3]

    matrixFileHandler.save(outfile.name, pSymmetric=True, pApplyCorrection=False)

    content_of_scool = cooler.fileops.list_scool_cells(outfile.name)
    content_expected = ['/cells/cell1', '/cells/cell2', '/cells/cell3']
    for content in content_expected:
        assert content in content_of_scool


def test_save_scool_pixeltables():
    outfile = NamedTemporaryFile(suffix='.scool', prefix='hicmatrix_scool_test')  # pylint: disable=R1732

    pMatrixFile = ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool051.cool'

    cooler_obj = cooler.Cooler(pMatrixFile)
    bins = cooler_obj.bins()[:]
    pixels = cooler_obj.pixels()[:]

    pixelsList = [pixels, pixels, pixels]
    matrices_list = ['cell1', 'cell2', 'cell3']
    matrixFileHandler = MatrixFileHandler(pFileType='scool')
    matrixFileHandler.matrixFile.coolObjectsList = None
    matrixFileHandler.matrixFile.bins = bins
    matrixFileHandler.matrixFile.pixel_list = pixelsList
    matrixFileHandler.matrixFile.name_list = matrices_list
    matrixFileHandler.save(outfile.name, pSymmetric=True, pApplyCorrection=False)

    content_of_scool = cooler.fileops.list_scool_cells(outfile.name)
    content_expected = ['/cells/cell1', '/cells/cell2', '/cells/cell3']
    for content in content_expected:
        assert content in content_of_scool


def test_load_cool_matrix_only():

    pMatrixFile = ROOT + 'GSE63525_GM12878_insitu_primary_2_5mb_hic2cool051.cool'

    matrixFileHandlerInput = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile, pLoadMatrixOnly=True)
    matrix, cut_intervals, nan_bins, \
        distance_counts, correction_factors = matrixFileHandlerInput.load()

    assert len(matrix) == 4
    assert cut_intervals is None
    assert nan_bins is None
    assert distance_counts is None
    assert correction_factors is None

    matrixFileHandlerInput2 = MatrixFileHandler(pFileType='cool', pMatrixFile=pMatrixFile)
    matrix2, _, _, \
        _, _ = matrixFileHandlerInput2.load()

    instances, features = matrix2.nonzero()
    nt.assert_almost_equal(matrix[0], instances, decimal=1)
    nt.assert_almost_equal(matrix[1], features, decimal=1)
    nt.assert_almost_equal(matrix[2], matrix2.data, decimal=1)
    assert matrix[3] == matrix2.shape[0]
