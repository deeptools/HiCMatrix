from hicmatrix.lib import MatrixFileHandler
import numpy.testing as nt
import os
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")
outfile = '/tmp/matrix'

# TODO(Frederic): add more testcases eg., more parameters for file handler, different parameters for saving ...


def test_load_and_save_homer(capsys):
    homer_outfile = outfile + '.homer'

    # create matrixFileHandler instance with filetype 'homer'
    pMatrixFile = ROOT + 'test_matrix.homer'
    fh = MatrixFileHandler(pFileType='homer', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    with pytest.raises(TypeError):
        fh.save(pName=homer_outfile, pSymmetric=False, pApplyCorrection=False)  # not implemented
        os.unlink(homer_outfile)


def test_load_and_save_h5():
    h5_outfile = outfile + '.h5'

    # create matrixFileHandler instance with filetype 'h5'
    pMatrixFile = ROOT + 'Li_et_al_2015.h5'
    fh = MatrixFileHandler(pFileType='h5', pMatrixFile=pMatrixFile)
    assert fh is not None

    # load data
    matrix, cut_intervals, nan_bins, distance_counts, correction_factors = fh.load()
    # set matrix variables
    fh.set_matrix_variables(matrix, cut_intervals, nan_bins, correction_factors, distance_counts)
    # and save it.
    fh.save(h5_outfile, True, None)

    os.unlink(h5_outfile)


def test_load_and_save_hicpro():
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
    with pytest.raises(TypeError):
        fh.save(pName=hicpro_outfile, pSymmetric=False, pApplyCorrection=False)  # not implemented
        os.unlink(hicpro_outfile)


def test_load_and_save_cool():
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


def test_load_and_save_ginteractions():
    pass
