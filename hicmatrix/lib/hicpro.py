
import logging
log = logging.getLogger(__name__)

from scipy.sparse import csr_matrix

from .matrixFile import MatrixFile


class Hicpro(MatrixFile, object):

    def __init__(self, pMatrixFile, pBedFile):
        super().__init__(pMatrixFileName=pMatrixFile, pBedFile=pBedFile)
        log.debug('pBedFile: {}'.format(pBedFile))

        log.debug('self.befile: {}'.format(self.bedFile))
    def load(self):
        instances = []
        features = []
        data = []
        with open(self.matrixFileName, 'r') as matrix_file:
            for line in matrix_file:
                x, y, value = line.strip().split('\t')
                instances.append(int(x) - 1)
                features.append(int(y) - 1)
                data.append(float(value))
        cut_intervals = []
        with open(self.bedFile, 'r') as bed_file:
            for line in bed_file:
                chrom, start, end, value = line.strip().split('\t')
                cut_intervals.append((chrom, int(start), int(end), int(value)))

        shape = len(cut_intervals)

        matrix = csr_matrix((data, (instances, features)), shape=(shape, shape))

        nan_bins = None
        distance_counts = None
        correction_factors = None
        return matrix, cut_intervals, nan_bins, distance_counts, correction_factors

    def save(self, pFilename, pSymmetric=None, pApplyCorrection=None):
        self.matrix.eliminate_zeros()
        instances, features = self.matrix.nonzero()
        data = self.matrix.data

        with open(pFilename, 'w') as matrix_file:
            for x, y, value in zip(instances, features, data):
                matrix_file.write(str(int(x+1)) + '\t' + str(int(y+1)) + '\t' + str(value) + '\n')
            
        with open(self.bedFile, 'w') as bed_file:
            for i, interval in enumerate(self.cut_intervals):
                bed_file.write('\t'.join(map(str, interval[:3])) + '\t' + str(i+1)+'\n')
        