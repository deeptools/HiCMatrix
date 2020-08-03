import os
import logging
log = logging.getLogger(__name__)
from datetime import datetime
from copy import deepcopy

import math
import time
import gc

import cooler
import h5py
import numpy as np
from scipy.sparse import triu, csr_matrix, lil_matrix, dok_matrix
import pandas as pd

from hicmatrix.utilities import toString, toBytes
from hicmatrix.utilities import convertNansToOnes, convertNansToZeros
from hicmatrix._version import __version__

from .matrixFile import MatrixFile


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
        self.hic_metadata = {}
        self.cool_info = None

        self.hic2cool_version = None
        self.hicmatrix_version = None
        self.distance = None
        self.matrixFormat = None
        self.matrixOnly = False
        self.noCutIntervals = False

    def getInformationCoolerBinNames(self):
        return cooler.Cooler(self.matrixFileName).bins().columns.values

    def load(self):
        log.debug('Load in cool format')
        self.minValue = None
        self.maxValue = None
        if self.matrixFileName is None:
            log.warning('No matrix is initialized')
        try:
            cooler_file = cooler.Cooler(self.matrixFileName)
            if 'metadata' in cooler_file.info:
                self.hic_metadata = cooler_file.info['metadata']
            else:
                self.hic_metadata = None
            self.cool_info = deepcopy(cooler_file.info)
        except Exception as e:
            log.warning("Could not open cooler file. Maybe the path is wrong or the given node is not available.")
            log.warning('The following file was tried to open: {}'.format(self.matrixFileName))
            log.warning("The following nodes are available: {}".format(cooler.fileops.list_coolers(self.matrixFileName.split("::")[0])))
            return None, e
        if self.chrnameList is None and (self.matrixFileName is None or not self.matrixOnly):
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
                del _data
                del _instances
                del _features

            if self.matrixFormat is None or self.matrixFormat == 'csr':
                matrix = csr_matrix((data, (instances, features)), shape=(np.int(cooler_file.info['nbins']), np.int(cooler_file.info['nbins'])), dtype=count_dtype)
            elif self.matrixFormat == 'lil':
                matrix = lil_matrix((data, (instances, features)), shape=(np.int(cooler_file.info['nbins']), np.int(cooler_file.info['nbins'])), dtype=count_dtype)
            elif self.matrixFormat == 'dok':
                matrix = dok_matrix((data, (instances, features)), shape=(np.int(cooler_file.info['nbins']), np.int(cooler_file.info['nbins'])), dtype=count_dtype)
            # elif  self.matrixFormat == 'raw':
            #     matrix = [instances, features, data, np.int(cooler_file.info['nbins'])]
            del data
            del instances
            del features
            gc.collect()
        elif self.chrnameList is None and self.matrixOnly:
            log.debug('Load all at once')
            matrixDataFrame = cooler_file.matrix(balance=False, sparse=True, as_pixels=True)
            used_dtype = np.int64
            # if np.iinfo(np.int32).max < cooler_file.info['nbins']:
            #     used_dtype = np.int64
            count_dtype = matrixDataFrame[0]['count'].dtype
            matrixDataFrameChunk = matrixDataFrame[:]
            data = matrixDataFrameChunk['count'].values.astype(count_dtype)
            instances = matrixDataFrameChunk['bin1_id'].values.astype(used_dtype)
            features = matrixDataFrameChunk['bin2_id'].values.astype(used_dtype)
            # matrix = [_instances, _features, _data, np.int(cooler_file.info['nbins'])]
            # return matrix, None, None, None, None
        else:
            if len(self.chrnameList) == 1:
                try:
                    if self.distance is None or cooler_file.binsize is None:
                        # load the full chromosome
                        matrix = cooler_file.matrix(balance=False, sparse=True, as_pixels=False).fetch(self.chrnameList[0]).tocsr()
                    else:
                        # load only the values up to a specific distance
                        lo, hi = cooler_file.extent(self.chrnameList[0])
                        dist = self.distance // cooler_file.binsize
                        step = (hi - lo) // 32
                        if step < 1:
                            step = 1
                        mat = lil_matrix((hi - lo, hi - lo), dtype=np.float32)

                        for i0, i1 in cooler.util.partition(lo, hi, step):
                            # fetch stripe
                            pixels = cooler_file.matrix(balance=False, as_pixels=True)[i0:i1, lo:hi]
                            # filter
                            pixels = pixels[(pixels['bin2_id'] - pixels['bin1_id']) < dist]
                            # insert into sparse matrix
                            mat[pixels['bin1_id'] - lo, pixels['bin2_id'] - lo] = pixels['count'].astype(np.float32)
                            del pixels

                        matrix = mat.tocsr()
                        del mat
                        gc.collect()

                except ValueError as ve:
                    log.exception("Wrong chromosome format. Please check UCSC / ensembl notation.")
                    log.exception('Error: {}'.format(str(ve)))
            else:
                raise Exception("Operation to load more as one region is not supported.")

        cut_intervals_data_frame = None
        correction_factors_data_frame = None

        if self.chrnameList is not None:
            if len(self.chrnameList) == 1:
                cut_intervals_data_frame = cooler_file.bins().fetch(self.chrnameList[0])
                log.debug('cut_intervals_data_frame {}'.format(list(cut_intervals_data_frame.columns)))
                if self.correctionFactorTable in cut_intervals_data_frame:
                    correction_factors_data_frame = cut_intervals_data_frame[self.correctionFactorTable]
            else:
                raise Exception("Operation to load more than one chr from bins is not supported.")
        else:
            if self.applyCorrectionLoad and self.correctionFactorTable in cooler_file.bins():
                correction_factors_data_frame = cooler_file.bins()[[self.correctionFactorTable]][:]

            cut_intervals_data_frame = cooler_file.bins()[['chrom', 'start', 'end']][:]

        correction_factors = None
        if correction_factors_data_frame is not None and self.applyCorrectionLoad:
            # apply correction factors to matrix
            # a_i,j = a_i,j * c_i *c_j
            if not self.matrixOnly:
                matrix.eliminate_zeros()
                data = matrix.data
            if len(data) > 1:

                if not self.matrixOnly:
                    matrix.data = matrix.data.astype(float)
                else:
                    data = np.array(data, dtype=float)

                correction_factors = np.array(correction_factors_data_frame.values).flatten()
                # Don't apply correction if weight were just 'nans'
                if np.sum(np.isnan(correction_factors)) != len(correction_factors):
                    # correction_factors = convertNansToZeros(correction_factors)

                    if not self.matrixOnly:
                        # matrix.sort_indices()
                        instances, features = matrix.nonzero()
                    instances_factors = correction_factors[instances]
                    features_factors = correction_factors[features]

                    if self.correctionOperator is None:
                        if self.correctionFactorTable in ['KR', 'VC', 'SQRT_VC']:
                            self.correctionOperator = '/'
                        else:
                            self.correctionOperator = '*'
                        if 'generated-by' in cooler_file.info:
                            log.debug('cooler_file.info[\'generated-by\'] {} {}'.format(cooler_file.info['generated-by'], type(cooler_file.info['generated-by'])))
                            generated_by = toString(cooler_file.info['generated-by'])
                            if 'hic2cool' in generated_by:
                                self.hic2cool_version = generated_by.split('-')[1]
                            elif 'hicmatrix' in generated_by:
                                self.hicmatrix_version = generated_by.split('-')[1]

                    instances_factors *= features_factors
                    log.debug('hic2cool: {}'.format(self.hic2cool_version))
                    log.debug('self.correctionOperator: {}'.format(self.correctionOperator))

                    if self.matrixOnly:
                        if self.correctionOperator == '*':
                            log.debug('multi')
                            data *= instances_factors
                        elif self.correctionOperator == '/':
                            log.debug('div')
                            data /= instances_factors
                        log.debug('non')
                        return [instances, features, data, np.int(cooler_file.info['nbins'])], None, None, None, None
                    else:
                        if self.correctionOperator == '*':
                            matrix.data *= instances_factors
                            log.debug('foo')
                        elif self.correctionOperator == '/':
                            matrix.data /= instances_factors
                            log.debug('hu')

        elif self.matrixOnly:
            return [instances, features, data, np.int(cooler_file.info['nbins'])], None, None, None, None

        cut_intervals = []
        if not self.noCutIntervals:
            for values in cut_intervals_data_frame.values:
                cut_intervals.append(tuple([toString(values[0]), values[1], values[2], 1.0]))
        del cut_intervals_data_frame
        del correction_factors_data_frame
        # try to restore nan_bins.
        try:
            # remove possible nan bins introduced by the correction factors
            # to have them part of the nan_bins vector
            mask = np.isnan(matrix.data)
            matrix.data[mask] = 0
            matrix.eliminate_zeros()
            shape = matrix.shape[0] if matrix.shape[0] < matrix.shape[1] else matrix.shape[1]
            nan_bins_indices = np.arange(shape)
            nan_bins_indices = np.setdiff1d(nan_bins_indices, matrix.indices)

            nan_bins = []
            for bin_id in nan_bins_indices:
                if len(matrix[bin_id, :].data) == 0:
                    nan_bins.append(bin_id)
            nan_bins = np.array(nan_bins)
        except Exception:
            nan_bins = None

        distance_counts = None

        return matrix, cut_intervals, nan_bins, distance_counts, correction_factors

    def create_cooler_input(self, pSymmetric=True, pApplyCorrection=True):
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

            # if the correction was applied by a division, invert it because cool format expects multiplicative if table name is 'weight'
            # https://cooler.readthedocs.io/en/latest/api.html#cooler.Cooler.matrix
            if (self.hic2cool_version is not None and self.hic2cool_version >= '0.5') or self.fileWasH5 or self.correctionOperator == '/':

                log.debug('h5 true')
                self.correction_factors = np.array(self.correction_factors).flatten()
                self.correction_factors = 1 / self.correction_factors
                mask = np.isnan(self.correction_factors)
                self.correction_factors[mask] = 0
                mask = np.isinf(self.correction_factors)
                self.correction_factors[mask] = 0
                self.correctionOperator = '*'
                log.debug('inverted correction factors')
            weight = convertNansToOnes(np.array(self.correction_factors).flatten())
            bins_data_frame = bins_data_frame.assign(weight=weight)

            log.debug("Reverting correction factors on matrix...")
            instances, features = self.matrix.nonzero()
            self.correction_factors = np.array(self.correction_factors)

            # do not apply if correction factors are just 1's
            instances_factors = self.correction_factors[instances]
            features_factors = self.correction_factors[features]

            instances_factors *= features_factors

            self.matrix.data = self.matrix.data.astype(float)

            # Apply the invert operation to get the original data
            if self.correctionOperator == '*' or self.correctionOperator is None:
                self.matrix.data /= instances_factors

            instances_factors = None
            features_factors = None

            self.matrix.eliminate_zeros()

        if self.correction_factors is not None and pApplyCorrection is False:
            dtype_pixel['weight'] = np.float32
            weight = convertNansToOnes(np.array(self.correction_factors).flatten())
            bins_data_frame = bins_data_frame.assign(weight=weight)

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
            log.debug("Writing non-standard cooler matrix. Datatype of matrix['count'] is: {}".format(self.matrix.dtype))
            dtype_pixel['count'] = self.matrix.dtype
        split_factor = 1
        if len(self.matrix.data) > 1e7:
            split_factor = 1e4
            matrix_data_frame = np.array_split(matrix_data_frame, split_factor)

        if self.appendData:
            self.appendData = 'a'
        else:
            self.appendData = 'w'

        info = {}
        # these fields are created by cooler lib. Can cause errors if not deleted.
        if 'metadata' in info:
            if self.hic_metadata is None:
                self.hic_metadata = info['metadata']
            del info['metadata']
        if 'bin-size' in info:
            del info['bin-size']
        if 'bin-type' in info:
            del info['bin-type']

        info['format'] = str('HDF5::Cooler')
        info['format-url'] = str('https://github.com/mirnylab/cooler')
        info['generated-by'] = str('HiCMatrix-' + __version__)
        info['generated-by-cooler-lib'] = str('cooler-' + cooler.__version__)

        info['tool-url'] = str('https://github.com/deeptools/HiCMatrix')

        # info['nchroms'] = int(bins_data_frame['chrom'][:].nunique())
        # info['chromosomes'] = list(bins_data_frame['chrom'][:].unique())
        # info['nnz'] = np.string_(str(self.matrix.nnz * 2))
        # info['min-value'] = np.string_(str(matrix_data_frame['count'].min()))
        # info['max-value'] = np.string_(str(matrix_data_frame['count'].max()))
        # info['sum-elements'] = int(matrix_data_frame['count'].sum())

        if self.hic_metadata is not None and 'matrix-generated-by' in self.hic_metadata:
            info['matrix-generated-by'] = str(self.hic_metadata['matrix-generated-by'])
            del self.hic_metadata['matrix-generated-by']
        if self.hic_metadata is not None and 'matrix-generated-by-url' in self.hic_metadata:
            info['matrix-generated-by-url'] = str(self.hic_metadata['matrix-generated-by-url'])
            del self.hic_metadata['matrix-generated-by-url']
        if self.hic_metadata is not None and 'genome-assembly' in self.hic_metadata:
            info['genome-assembly'] = str(self.hic_metadata['genome-assembly'])
            del self.hic_metadata['genome-assembly']

        return bins_data_frame, matrix_data_frame, dtype_pixel, info

    def save(self, pFileName, pSymmetric=True, pApplyCorrection=True):
        log.debug('Save in cool format')

        bins_data_frame, matrix_data_frame, dtype_pixel, info = self.create_cooler_input(pSymmetric=pSymmetric, pApplyCorrection=pApplyCorrection)
        local_temp_dir = os.path.dirname(os.path.realpath(pFileName))
        cooler.create_cooler(cool_uri=pFileName,
                             bins=bins_data_frame,
                             pixels=matrix_data_frame,
                             mode=self.appendData,
                             dtypes=dtype_pixel,
                             ordered=True,
                             metadata=self.hic_metadata,
                             temp_dir=local_temp_dir)

        if self.appendData == 'w':
            fileName = pFileName.split('::')[0]
            with h5py.File(fileName, 'r+') as h5file:
                h5file.attrs.update(info)
                h5file.close()
