import cooler
import logging
import numpy as np
from scipy.sparse import triu, csr_matrix
import pandas as pd
from past.builtins import zip
from builtins import super
log = logging.getLogger(__name__)
from .matrixFile import MatrixFile
from hicmatrix.utilities import toString
from hicmatrix.utilities import convertNansToOnes


class Cool(MatrixFile, object):

    def __init__(self, pMatrixFile=None):
        super().__init__(pMatrixFile)
        self.chrnameList = None
        self.correctionFactorTable = 'weight'
        self.correctionOperator = '*'
        self.enforceInteger = False
        self.appendData = False
        self.fileWasH5 = False
        self.applyCorrectionLoad = True

    def getInformationCoolerBinNames(self):
        return cooler.Cooler(self.matrixFileName).bins().columns.values

    def load(self):
        log.debug('Load in cool format')
        if self.matrixFileName is None:
            log.info('No matrix is initalized')

        try:
            cooler_file = cooler.Cooler(self.matrixFileName)
        except Exception:
            log.info("Could not open cooler file. Maybe the path is wrong or the given node is not available.")
            log.info('The following file was tried to open: {}'.format(self.matrixFileName))
            log.info("The following nodes are available: {}".format(cooler.io.ls(self.matrixFileName.split("::")[0])))
            exit()

        if self.chrnameList is None:
            matrixDataFrame = cooler_file.matrix(balance=False, sparse=True, as_pixels=True)
            used_dtype = np.int32
            if np.iinfo(np.int32).max < cooler_file.info['nbins']:
                used_dtype = np.int64
            count_dtype = matrixDataFrame[0]['count'].dtype
            data = np.empty(cooler_file.info['nnz'], dtype=count_dtype)
            instances = np.empty(cooler_file.info['nnz'], dtype=used_dtype)
            features = np.empty(cooler_file.info['nnz'], dtype=used_dtype)
            i = 0
            size = cooler_file.info['nbins'] // 32
            if size == 0:
                size = 1
            start_pos = 0
            while i < cooler_file.info['nbins']:
                _data = matrixDataFrame[i:i + size]['count'].values.astype(count_dtype).T
                _instances = matrixDataFrame[i:i + size]['bin1_id'].values.astype(used_dtype).T
                _features = matrixDataFrame[i:i + size]['bin2_id'].values.astype(used_dtype).T

                data[start_pos:start_pos + len(_data)] = _data
                instances[start_pos:start_pos + len(_instances)] = _instances
                features[start_pos:start_pos + len(_features)] = _features
                start_pos += len(_features)
                i += size

            matrix = csr_matrix((data, (instances, features)), shape=(cooler_file.info['nbins'], cooler_file.info['nbins']), dtype=count_dtype)
            # del data
            # del instances
            # del features
        else:
            if len(self.chrnameList) == 1:
                try:
                    matrix = cooler_file.matrix(balance=False, sparse=True).fetch(self.chrnameList[0]).tocsr()
                except ValueError:
                    exit("Wrong chromosome format. Please check UCSC / ensembl notation.")
            else:
                exit("Operation to load more as one region is not supported.")

        cut_intervals_data_frame = None
        correction_factors_data_frame = None

        if self.chrnameList is not None:
            if len(self.chrnameList) == 1:
                cut_intervals_data_frame = cooler_file.bins().fetch(self.chrnameList[0])

                if self.correctionFactorTable in cut_intervals_data_frame:
                    correction_factors_data_frame = cut_intervals_data_frame[self.correctionFactorTable]
            else:
                exit("Operation to load more than one chr from bins is not supported.")
        else:
            if self.applyCorrectionLoad and self.correctionFactorTable in cooler_file.bins():
                correction_factors_data_frame = cooler_file.bins()[[self.correctionFactorTable]][:]

            cut_intervals_data_frame = cooler_file.bins()[['chrom', 'start', 'end']][:]

        correction_factors = None
        if correction_factors_data_frame is not None and self.applyCorrectionLoad:
            # apply correction factors to matrix
            # a_i,j = a_i,j * c_i *c_j
            matrix.eliminate_zeros()
            matrix.data = matrix.data.astype(float)

            correction_factors = convertNansToOnes(np.array(correction_factors_data_frame.values).flatten())
            # apply only if there are not only 1's
            if np.sum(correction_factors) != len(correction_factors):
                matrix.sort_indices()

                instances, features = matrix.nonzero()
                instances_factors = correction_factors[instances]
                features_factors = correction_factors[features]
                instances_factors *= features_factors
                log.debug('Apply {}'.format(self.correctionOperator))
                if self.correctionOperator == '*':
                    matrix.data *= instances_factors
                elif self.correctionOperator == '/':
                    matrix.data /= instances_factors

        cut_intervals = []

        for values in cut_intervals_data_frame.values:
            cut_intervals.append(tuple([toString(values[0]), values[1], values[2], 1.0]))

        # try to restore nan_bins.
        try:
            shape = matrix.shape[0] if matrix.shape[0] < matrix.shape[1] else matrix.shape[1]
            nan_bins = np.arange(shape)
            nan_bins = np.setdiff1d(nan_bins, matrix.indices[:-1])

        except Exception:
            nan_bins = None

        distance_counts = None

        return matrix, cut_intervals, nan_bins, distance_counts, correction_factors

    def save(self, pFileName, pSymmetric=True, pApplyCorrection=True):
        log.debug('Save in cool format')

        self.matrix.eliminate_zeros()

        if self.nan_bins is not None and len(self.nan_bins) > 0 and self.fileWasH5:
            # remove nan_bins
            correction_factors = np.ones(self.matrix.shape[0])
            correction_factors[self.nan_bins] = 0
            self.matrix.sort_indices()
            _instances, _features = self.matrix.nonzero()

            instances_factors = correction_factors[_instances]
            features_factors = correction_factors[_features]

            instances_factors = np.logical_not(np.logical_or(instances_factors, features_factors))
            self.matrix.data[instances_factors] = 0
            self.matrix.eliminate_zeros()

        # set possible nans in data to 0
        mask = np.isnan(self.matrix.data)

        self.matrix.data[mask] = 0
        self.matrix.eliminate_zeros()
        # save only the upper triangle of the
        if pSymmetric:
            # symmetric matrix
            self.matrix = triu(self.matrix, format='csr')
        else:
            self.matrix = self.matrix

        self.matrix.eliminate_zeros()

        # create data frame for bins
        # self.cut_intervals is having 4 tuples, bin_data_frame should have 3.correction_factors
        # it looks like it is faster to create it with 4, and drop the last one
        # instead of handling this before.
        bins_data_frame = pd.DataFrame(self.cut_intervals, columns=['chrom', 'start', 'end', 'interactions']).drop('interactions', axis=1)
        dtype_pixel = {'bin1_id': np.int32, 'bin2_id': np.int32, 'count': np.int32}

        if self.correction_factors is not None and pApplyCorrection:
            dtype_pixel['weight'] = np.float32
            if self.fileWasH5:
                self.correction_factors = np.array(self.correction_factors).flatten()
                self.correction_factors = 1 / self.correction_factors
                mask = np.isnan(self.correction_factors)
                self.correction_factors[mask] = 0
                mask = np.isinf(self.correction_factors)
                self.correction_factors[mask] = 0
            weight = convertNansToOnes(np.array(self.correction_factors).flatten())
            bins_data_frame = bins_data_frame.assign(weight=weight)

            log.info("Reverting correction factors on matrix...")
            instances, features = self.matrix.nonzero()
            self.correction_factors = np.array(self.correction_factors)

            # do not apply if correction factors are just 1's
            instances_factors = self.correction_factors[instances]
            features_factors = self.correction_factors[features]

            instances_factors *= features_factors
            self.matrix.data = self.matrix.data.astype(float)

            # Apply the invert operation to get the original data
            if self.correctionOperator == '*':
                self.matrix.data /= instances_factors
            elif self.correctionOperator == '/':
                self.matrix.data *= instances_factors

            instances_factors = None
            features_factors = None

            self.matrix.eliminate_zeros()

        instances, features = self.matrix.nonzero()

        matrix_data_frame = pd.DataFrame(instances, columns=['bin1_id'], dtype=np.int32)
        del instances
        matrix_data_frame = matrix_data_frame.assign(bin2_id=features)
        del features

        if self.enforceInteger:
            dtype_pixel['count'] = np.int32
            data = np.rint(self.matrix.data)
            matrix_data_frame = matrix_data_frame.assign(count=data)
        else:
            matrix_data_frame = matrix_data_frame.assign(count=self.matrix.data)

        if not self.enforceInteger and self.matrix.dtype not in [np.int32, int]:
            log.warning("Writing non-standard cooler matrix. Datatype of matrix['count'] is: {}".format(self.matrix.dtype))
            dtype_pixel['count'] = self.matrix.dtype
        split_factor = 1
        if len(self.matrix.data) > 1e7:
            split_factor = 1e4
            matrix_data_frame = np.array_split(matrix_data_frame, split_factor)

        cooler.io.create(cool_uri=pFileName,
                         bins=bins_data_frame,
                         pixels=matrix_data_frame,
                         append=self.appendData,
                         dtype=dtype_pixel)
