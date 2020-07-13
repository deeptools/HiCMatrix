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


class Scool(MatrixFile, object):

    def __init__(self, pMatrixFile=None):
        super().__init__(pMatrixFile)
        log.debug('scool object created')
        self.coolObjectsList = None

    def load(self):
        raise NotImplementedError('Please use the specific cell to load the individual cool file from the scool file')
        exit(1)
    
    def save(self, pFileName, pSymmetric=True, pApplyCorrection=True):

        pixel_dict = {}
        bins_dict = {}
        for coolObject in self.coolObjectsList:
            bins_data_frame, matrix_data_frame, dtype_pixel, info = coolObject.matrixFile.create_cooler_input(pSymmetric=pSymmetric, pApplyCorrection=pApplyCorrection)
            # print('key name: {}'.format(coolObject.matrixFile.matrixFileName))
            bins_dict[coolObject.matrixFile.matrixFileName] = bins_data_frame
            pixel_dict[coolObject.matrixFile.matrixFileName] = matrix_data_frame

            # pixel_list.append(matrix_data_frame)
            # cell_names_list.append(coolObject.matrixFile.matrixFileName)
        
        local_temp_dir = os.path.dirname(os.path.realpath(pFileName))
        # log.debug('pFileName {}'.format(pFileName))
        # log.debug('bins_data_frame {}'.format(bins_data_frame))
        # log.debug('pixel_list {}'.format(pixel_list[:2]))
        # log.debug('cell_names_list {}'.format(cell_names_list[:2]))

# cool_uri, bins_dict, cell_name_pixels_dict
        cooler.create_scool(cool_uri=pFileName, bins_dict=bins_dict, cell_name_pixels_dict=pixel_dict,
                            dtypes=dtype_pixel,
                            ordered=True,
                            temp_dir=local_temp_dir)