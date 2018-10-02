from .matrixFile import MatrixFile
from scipy.sparse import csr_matrix
from builtins import super
import gzip
import logging
log = logging.getLogger(__name__)


class Homer(MatrixFile, object):

    def __init__(self, pMatrixFile):
        super().__init__(pMatrixFile)

    def load(self):
        cut_intervals = []

        with open(self.matrixFileName, 'r') as matrix_file:
            values = matrix_file.readline()
            values = values.strip().split('\t')

            # get bin size
            start_first = int(values[2].strip().split('-')[1])
            start_second = int(values[3].strip().split('-')[1])
            bin_size = start_second - start_first
            for i, value in enumerate(values[2:]):
                chrom, start = value.strip().split('-')
                cut_intervals.append((chrom, int(start), int(start) + bin_size, 1))

            matrix_dense = []
            for line in matrix_file:
                values = line.split('\t')
                data = []
                for i, value in enumerate(values[2:]):
                    data.append(float(value))
                matrix_dense.append(data)

        matrix = csr_matrix(matrix_dense)
        nan_bins = None
        distance_counts = None
        correction_factors = None
        return matrix, cut_intervals, nan_bins, distance_counts, correction_factors

    def save(self, pFilename, pSymmetric=None, pApplyCorrection=None):

        with gzip.open(pFilename, 'wt') as homerMatrixFile:
            homerMatrixFile.write('HiCMatrix (directory=.)\tRegions\t')
            for bin_interval in self.cut_intervals:
                homerMatrixFile.write('{}-{}\t'.format(bin_interval[0], bin_interval[1]))
            homerMatrixFile.write('\n')

            for i in range(self.matrix.shape[0]):
                data = '\t'.join(map(str, self.matrix[i, :].toarray().flatten()))
                homerMatrixFile.write('{}-{}\t{}-{}\t'.format(self.cut_intervals[i][0], self.cut_intervals[i][1],
                                                              self.cut_intervals[i][0], self.cut_intervals[i][1]))
                homerMatrixFile.write('{}'.format(data))
                if i < self.matrix.shape[0] - 1:
                    homerMatrixFile.write('\n')
