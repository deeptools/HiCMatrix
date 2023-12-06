import logging

import gzip

from scipy.sparse import csr_matrix

from hicmatrix.utilities import opener

from .matrixFile import MatrixFile

log = logging.getLogger(__name__)


class Homer(MatrixFile):

    def __init__(self, pMatrixFile):
        super().__init__(pMatrixFile)

    def load(self):
        cut_intervals = []

        # matrix_file = opener(self.matrixFileName)
        with opener(self.matrixFileName) as matrix_file:
            values = matrix_file.readline()
            values = values.strip().split(b'\t')

            # get bin size
            start_first = int(values[2].strip().split(b'-')[1])
            start_second = int(values[3].strip().split(b'-')[1])
            bin_size = start_second - start_first
            for value in values[2:]:
                chrom, start = value.strip().split(b'-')
                cut_intervals.append((chrom.decode('ascii'), int(start), int(start) + bin_size, 1))

            matrix_dense = []
            for line in matrix_file:
                values = line.split(b'\t')
                data = []
                for value in values[2:]:
                    data.append(float(value))
                matrix_dense.append(data)
        # matrix_file.close()
        matrix = csr_matrix(matrix_dense)
        nan_bins = None
        distance_counts = None
        correction_factors = None
        return matrix, cut_intervals, nan_bins, distance_counts, correction_factors

    def save(self, pFileName, pSymmetric=None, pApplyCorrection=None):

        with gzip.open(pFileName, 'wt') as homerMatrixFile:
            homerMatrixFile.write('HiCMatrix (directory=.)\tRegions\t')
            for bin_interval in self.cut_intervals:
                homerMatrixFile.write(f'{bin_interval[0]}-{bin_interval[1]}\t')
            homerMatrixFile.write('\n')

            for i in range(self.matrix.shape[0]):
                data = '\t'.join(map(str, self.matrix[i, :].toarray().flatten()))
                homerMatrixFile.write(f'{self.cut_intervals[i][0]}-{self.cut_intervals[i][1]}\t{self.cut_intervals[i][0]}-{self.cut_intervals[i][1]}\t')
                homerMatrixFile.write(f'{data}')
                if i < self.matrix.shape[0] - 1:
                    homerMatrixFile.write('\n')
