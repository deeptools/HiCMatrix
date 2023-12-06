import logging
import os

log = logging.getLogger(__name__)
import gc
import math
import time
from copy import deepcopy
from datetime import datetime

import cooler
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix, triu

from hicmatrix.utilities import (convertNansToOnes, convertNansToZeros,
                                 toBytes, toString)

from .matrixFile import MatrixFile


class Scool(MatrixFile, object):

    def __init__(self, pMatrixFile=None):
        super().__init__(pMatrixFile)
        log.debug('scool object created')
        self.coolObjectsList = None
        self.bins = None
        self.pixel_list = None
        self.name_list = None

    def load(self):
        raise NotImplementedError('Please use the specific cell to load the individual cool file from the scool file')
        exit(1)

    def save(self, pFileName, pSymmetric=True, pApplyCorrection=True):

        pixel_dict = {}
        bins_dict = {}

        if self.coolObjectsList is not None:
            for coolObject in self.coolObjectsList:
                bins_data_frame, matrix_data_frame, dtype_pixel, info = coolObject.matrixFile.create_cooler_input(pSymmetric=pSymmetric, pApplyCorrection=pApplyCorrection)
                bins_dict[coolObject.matrixFile.matrixFileName] = bins_data_frame
                pixel_dict[coolObject.matrixFile.matrixFileName] = matrix_data_frame

        else:
            try:
                dtype_pixel = {'bin1_id': np.int32, 'bin2_id': np.int32, 'count': self.pixel_list[0]['count'].dtype}
                # dtype_pixel = self.pixel_list[0]['count'].dtype

                for i, pixels in enumerate(self.pixel_list):
                    bins_dict[self.name_list[i]] = self.bins
                    pixel_dict[self.name_list[i]] = pixels
                    log.debug('self.name_list[i] {}'.format(self.name_list[i]))
            except Exception as exp:
                log.debug('Exception {}'.format(str(exp)))

        local_temp_dir = os.path.dirname(os.path.realpath(pFileName))

        cooler.create_scool(cool_uri=pFileName, bins=bins_dict, cell_name_pixels_dict=pixel_dict,
                            dtypes=dtype_pixel,
                            ordered=True,
                            temp_dir=local_temp_dir)
