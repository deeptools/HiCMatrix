import os
import cooler
import logging
import datetime

import numpy as np
from scipy.sparse import triu, csr_matrix
import pandas as pd
from past.builtins import zip
from builtins import super
log = logging.getLogger(__name__)
from .matrixFile import MatrixFile
from hicmatrix.utilities import toString
from hicmatrix.utilities import convertNansToOnes
from hicmatrix._version import __version__


class Cool(MatrixFile, object):

    def __init__(self, pMatrixFile=None):
        super().__init__(pMatrixFile)
        self.chrnameList = None
        self.correctionFactorTable = 'weight'
        self.correctionOperator = None
        self.enforceInteger = False
        self.appendData = False
        self.fileWasH5 = False
        self.applyCorrectionLoad = True
        self.hic_info = {}
        self.hic2cool_version = None
        self.hicmatrix_version = None

    def getInformationCoolerBinNames(self):
        return cooler.Cooler(self.matrixFileName).bins().columns.values

    def load(self):
        log.debug('Load in cool format')
        if self.matrixFileName is None:
            log.info('No matrix is initalized')

        try:
            cooler_file = cooler.Cooler(self.matrixFileName)
            self.hic_info = cooler_file.info
        except Exception:
            log.info("Could not open cooler file. Maybe the path is wrong or the given node is not available.")
            log.info('The following file was tried to open: {}'.format(self.matrixFileName))
            log.info("The following nodes are available: {}".format(cooler.fileops.list_coolers(self.matrixFileName.split("::")[0])))
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
                matrixDataFrameChunk = matrixDataFrame[i:i + size]
                _data = matrixDataFrameChunk['count'].values.astype(count_dtype)
                _instances = matrixDataFrameChunk['bin1_id'].values.astype(used_dtype)
                _features = matrixDataFrameChunk['bin2_id'].values.astype(used_dtype)

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

                if self.correctionOperator is None:
                    if 'generated-by' in cooler_file.info:
                        generated_by = cooler_file.info['generated-by']
                        if 'hic2cool' in generated_by:
                            self.hic2cool_version = generated_by.split('-')[1]
                            if self.hic2cool_version <= '0.5':
                                self.correctionOperator = '*'
                            else:
                                self.correctionOperator = '/'
                        elif 'hicmatrix' in generated_by:
                            self.hicmatrix_version = generated_by.split('-')[1]
                            if self.hicmatrix_version <= '7':
                                self.correctionOperator = '*'
                            else:
                                self.correctionOperator = '/'
                    else:
                        self.correctionOperator = '*'
                log.debug('Apply {}'.format(self.correctionOperator))

                if self.hic2cool_version is not None and self.hic2cool_version >= '0.5':
                    instances_factors /= features_factors
                elif self.hicmatrix_version is not None and self.hicmatrix_version >= '8':
                    instances_factors /= features_factors
                else:
                    instances_factors *= features_factors
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

            if self.hic2cool_version is not None and self.hic2cool_version >= '0.5':
                instances_factors /= features_factors
            elif self.hicmatrix_version is not None and self.hicmatrix_version >= '8':
                instances_factors /= features_factors
            else:
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

        if self.appendData:
            self.appendData = 'a'
        else:
            self.appendData = 'w'

        if bins_data_frame['start'][2] - bins_data_frame['start'][1] == bins_data_frame['start'][12] - bins_data_frame['start'][11]:
            # bin_size = bins_data_frame['start'].mean
            self.hic_info['bin-size'] = int(bins_data_frame['start'][2] - bins_data_frame['start'][1])
            self.hic_info['bin-type'] = 'fixed'
        else:
            self.hic_info['bin-size'] = None
            self.hic_info['bin-type'] = 'variable'
        self.hic_info['creation-date'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
        self.hic_info['format'] = 'HDF5::Cooler'
        self.hic_info['format-url'] = 'https://github.com/mirnylab/cooler'
        self.hic_info['generated-by'] = 'HiCMatrix-' + __version__
        self.hic_info['generated-by-cooler-lib'] = 'cooler-' + cooler.__version__

        self.hic_info['tool-url'] = 'https://github.com/deeptools/HiCMatrix'

        # self.hic_info['genome-assembly'] = ''
        if 'graphs' in self.hic_info:
            del self.hic_info['graphs']
        self.hic_info['nbins'] = int(self.matrix.shape[0])
        self.hic_info['nchroms'] = int(bins_data_frame['chrom'][:].nunique())
        self.hic_info['chromosomes'] = list(bins_data_frame['chrom'][:].unique())
        self.hic_info['nnz'] = int(self.matrix.nnz) * 2
        self.hic_info['min-value'] = int(matrix_data_frame['count'].min())
        self.hic_info['max-value'] = int(matrix_data_frame['count'].max())
        self.hic_info['sum-elements'] = int(matrix_data_frame['count'].sum())

        if 'statistics' in self.hic_info:
            log.debug('statistics {}'.format(self.hic_info['statistics']))
        else:
            log.debug('no stats')
            # self.hic_info['statistics'] = ''

        local_temp_dir = os.path.dirname(os.path.realpath(pFileName))
        cooler.create_cooler(cool_uri=pFileName,
                             bins=bins_data_frame,
                             pixels=matrix_data_frame,
                             mode=self.appendData,
                             dtypes=dtype_pixel,
                             ordered=True,
                             metadata=self.hic_info,
                             temp_dir=local_temp_dir)
