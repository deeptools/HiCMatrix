from hicmatrix.lib import MatrixFileHandler
import numpy.testing as nt
import os
import pytest
from scipy.sparse.csr import csr_matrix
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")
outfile = '/tmp/matrix'

# TODO(Frederic): add more testcases eg., more parameters for file handler, different parameters for saving ...


def test_load_homer(capsys):
    # create matrixFileHandler instance with filetype 'homer'
    pMatrixFile = ROOT + 'test_matrix.homer'
    fh = MatrixFileHandler(pFileType='homer', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()

    # TODO(Frederic): figure out how matrix can be tested: matrix[0] prints values as below of type csr_matrix

    # test_matrix = np.array([[(0, 0), 1.0],
    #                         [(0, 1), 0.1896],
    #                         [(0, 2), 0.2163],
    #                         [(0, 3), 0.08288],
    #                         [(0, 4), 0.1431],
    #                         [(0, 5), 0.2569],
    #                         [(0, 6), 0.1315],
    #                         [(0, 7), 0.1488],
    #                         [(0, 8), -0.0312],
    #                         [(0, 9), 0.143],
    #                         [(0, 10), 0.06091],
    #                         [(0, 11), 0.03546],
    #                         [(0, 12), 0.1168]])

    # test_csr = csr_matrix(test_matrix)
    # nt.assert_almost_equal(matrix[0], test_csr)

    test_cut_intervals = [('3R', 1000000, 1020000, 1), ('3R', 1020000, 1040000, 1), ('3R', 1040000, 1060000, 1), ('3R', 1060000, 1080000, 1), ('3R', 1080000, 1100000, 1), ('3R', 1100000, 1120000, 1), ('3R', 1120000, 1140000, 1), ('3R', 1140000, 1160000, 1), ('3R', 1160000, 1180000, 1), ('3R', 1180000, 1200000, 1), ('3R', 1200000, 1220000, 1), ('3R', 1220000, 1240000, 1), ('3R', 1240000, 1260000, 1)]  # noqa E501
    nt.assert_equal(cut_intervals, test_cut_intervals)

    assert nan_bins is None
    assert distance_counts is None
    assert correction_factors is None


@pytest.mark.xfail
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

    # TODO(Frederic): figure out how to test matrix. first printed values are below.
    # (8, 8)        1054.4748299674175
    # (8, 9)        375.86990248276913
    # (8, 10)       222.53900279141928
    # (8, 11)       114.26339810150536
    # (8, 12)       95.87462677732627
    # (8, 13)       58.183562077292635
    # (8, 14)       40.02820075083617
    # (8, 15)       41.80159709758394
    # (8, 16)       42.69569769771059
    # (8, 17)       52.34881059467604
    # (8, 18)       47.88271388848242
    # (8, 19)       42.88766437656768
    # (8, 20)       42.0588857716962
    # (8, 21)       34.58304559227142
    # (8, 22)       25.402568709457164
    # (8, 23)       24.117355326662004
    # (8, 24)       26.679125756964773
    # (8, 25)       13.37324675200762
    # (8, 26)       24.52909568542892
    # (8, 27)       34.41541621515177
    # (8, 28)       29.7791587382894
    # (8, 29)       23.326878297248342
    # (8, 33)       30.8959743764029
    # (8, 34)       23.889036207055536
    # (8, 35)       21.351014711942522

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

    # TODO(Frederic): figure out how to test matrix.
    # (0, 0)	41.345793
    # (0, 827)	5.42079
    # (0, 1263)	5.122642
    # (1, 1)	21.999631
    # (1, 2)	12.420673
    # (1, 16)	3.42396
    # (1, 17)	3.036948
    # (1, 41)	2.947819
    # (1, 43)	2.19871
    # (1, 211)	2.295818
    # (1, 224)	2.043807
    # (1, 2823)	1.521666
    # (2, 2)	3.895851
    # (2, 3)	7.810459
    # (2, 4)	2.553446
    # (2, 5)	2.19859
    # (2, 6)	3.069612
    # (2, 8)	2.540354
    # (2, 10)	2.203212
    # (2, 11)	4.614919
    # (2, 20)	2.127198
    # (2, 57)	1.536444
    # (2, 84)	1.382808
    # (2, 226)	1.993518
    # (2, 1833)	1.609879

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

    # TODO(Frederic): figure out how to test matrix.

    # (8, 8)	713.5520292527104
    # (8, 9)	486.2172222094596
    # (8, 10)	272.6487169310663
    # (8, 11)	128.32819244904616
    # (8, 12)	83.46301425413387
    # (8, 13)	47.00026389581942
    # (8, 14)	27.31375604270838
    # (8, 15)	24.535696066366036
    # (8, 16)	37.71746991217174
    # (8, 17)	32.230802046781434
    # (8, 18)	40.53231811826729
    # (8, 19)	43.32588229935525
    # (8, 20)	33.068670586367574
    # (8, 21)	14.078448491840975
    # (8, 22)	2.6942558141809667
    # (8, 23)	8.516902198953957
    # (8, 24)	4.5411139319573035
    # (8, 25)	1.9744505067257068
    # (8, 26)	3.3255396538666138
    # (8, 27)	16.823860295768796
    # (8, 28)	34.73491038681628
    # (8, 29)	15.855620883345953
    # (8, 33)	3.5281952793284255
    # (8, 34)	10.867553045286272
    # (8, 35)	42.44332645326515

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


def test_load_ginteractions():
    pass


def test_save_ginteractions():
    pass
