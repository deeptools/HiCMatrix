import logging
import os
from os import unlink

import numpy as np
import tables
from scipy.sparse import csr_matrix, triu

from hicmatrix.utilities import toString

from .matrixFile import MatrixFile

log = logging.getLogger(__name__)


class H5(MatrixFile):

    def __init__(self, pMatrixFile):
        super().__init__(pMatrixFile)

    def load(self):
        """
        Loads a matrix stored in h5 format
        :param matrix_filename:
        :return: matrix, cut_intervals, nan_bins, distance_counts, correction_factors
        """
        log.debug('Load in h5 format')

        with tables.open_file(self.matrixFileName, 'r') as f:
            parts = {}
            try:
                for matrix_part in ('data', 'indices', 'indptr', 'shape'):
                    parts[matrix_part] = getattr(f.root.matrix, matrix_part).read()
            except Exception as e:  # pylint: disable=W0718
                log.info('No h5 file. Please check parameters concerning the file type!')
                # Should probably be raise e:
                e  # pylint: disable=W0104
            matrix = csr_matrix(tuple([parts['data'], parts['indices'], parts['indptr']]),
                                shape=parts['shape'])
            # matrix = hiCMatrix.fillLowerTriangle(matrix)
            # get intervals
            intvals = {}
            for interval_part in ('chr_list', 'start_list', 'end_list', 'extra_list'):
                if toString(interval_part) == toString('chr_list'):
                    chrom_list = getattr(f.root.intervals, interval_part).read()
                    intvals[interval_part] = toString(chrom_list)
                else:
                    intvals[interval_part] = getattr(f.root.intervals, interval_part).read()

            cut_intervals = list(zip(intvals['chr_list'], intvals['start_list'], intvals['end_list'], intvals['extra_list']))
            assert len(cut_intervals) == matrix.shape[0], \
                f"Error loading matrix. Length of bin intervals ({len(cut_intervals)}) is different than the " \
                f"size of the matrix ({matrix.shape[0]})"

            # get nan_bins
            try:
                if hasattr(f.root, 'nan_bins'):
                    nan_bins = f.root.nan_bins.read()
                else:
                    nan_bins = np.array([])
            except Exception:  # pylint: disable=W0718
                nan_bins = np.array([])

            # get correction factors
            try:
                if hasattr(f.root, 'correction_factors'):
                    correction_factors = f.root.correction_factors.read()
                    assert len(correction_factors) == matrix.shape[0], \
                        "Error loading matrix. Length of correction factors does not" \
                        "match size of matrix"
                    correction_factors = np.array(correction_factors)
                    mask = np.isnan(correction_factors)
                    correction_factors[mask] = 0
                    mask = np.isinf(correction_factors)
                    correction_factors[mask] = 0
                else:
                    correction_factors = None
            except Exception:  # pylint: disable=W0718
                correction_factors = None

            try:
                # get correction factors
                if hasattr(f.root, 'distance_counts'):
                    distance_counts = f.root.correction_factors.read()
                else:
                    distance_counts = None
            except Exception:  # pylint: disable=W0718
                distance_counts = None
            return matrix, cut_intervals, nan_bins, distance_counts, correction_factors

    def save(self, pFileName, pSymmetric=True, pApplyCorrection=None):
        """
        Saves a matrix using hdf5 format
        :param pFileName:
        :return: None
        """
        log.debug('Save in h5 format')

        # self.restoreMaskedBins()
        if not pFileName.endswith(".h5"):
            pFileName += ".h5"

        # if the file name already exists
        # try to find a new suitable name
        if os.path.isfile(pFileName):
            log.warning("*WARNING* File already exists %s\n "
                        "Overwriting ...\n", pFileName)

            unlink(pFileName)
        if self.nan_bins is None:
            self.nan_bins = np.array([])
        elif not isinstance(self.nan_bins, np.ndarray):
            self.nan_bins = np.array(self.nan_bins)

        # save only the upper triangle of the
        if pSymmetric:
            # symmetric matrix
            matrix = triu(self.matrix, k=0, format='csr')
        else:
            matrix = self.matrix
        matrix.eliminate_zeros()

        filters = tables.Filters(complevel=5, complib='blosc')
        with tables.open_file(pFileName, mode="w", title="HiCExplorer matrix") as h5file:
            matrix_group = h5file.create_group("/", "matrix", )
            # save the parts of the csr matrix
            for matrix_part in ('data', 'indices', 'indptr', 'shape'):
                arr = np.array(getattr(matrix, matrix_part))
                atom = tables.Atom.from_dtype(arr.dtype)
                ds = h5file.create_carray(matrix_group, matrix_part, atom,
                                          shape=arr.shape,
                                          filters=filters)
                ds[:] = arr

            # save the matrix intervals
            intervals_group = h5file.create_group("/", "intervals", )
            chr_list, start_list, end_list, extra_list = zip(*self.cut_intervals)  # pylint: disable=W0612
            for interval_part in ('chr_list', 'start_list', 'end_list', 'extra_list'):
                arr = np.array(eval(interval_part))  # pylint: disable=W0123
                atom = tables.Atom.from_dtype(arr.dtype)
                ds = h5file.create_carray(intervals_group, interval_part, atom,
                                          shape=arr.shape,
                                          filters=filters)
                ds[:] = arr

            # save nan bins
            if len(self.nan_bins):
                atom = tables.Atom.from_dtype(self.nan_bins.dtype)
                ds = h5file.create_carray(h5file.root, 'nan_bins', atom,
                                          shape=self.nan_bins.shape,
                                          filters=filters)
                ds[:] = self.nan_bins

            # save corrections factors
            if self.correction_factors is not None and len(self.correction_factors):
                self.correction_factors = np.array(self.correction_factors)
                mask = np.isnan(self.correction_factors)
                self.correction_factors[mask] = 0
                atom = tables.Atom.from_dtype(self.correction_factors.dtype)
                ds = h5file.create_carray(h5file.root, 'correction_factors', atom,
                                          shape=self.correction_factors.shape,
                                          filters=filters)
                ds[:] = np.array(self.correction_factors)

            # save distance counts
            if self.distance_counts is not None and len(self.distance_counts):
                atom = tables.Atom.from_dtype(self.distance_counts.dtype)
                ds = h5file.create_carray(h5file.root, 'distance_counts', atom,
                                          shape=self.distance_counts.shape,
                                          filters=filters)
                ds[:] = np.array(self.distance_counts)
