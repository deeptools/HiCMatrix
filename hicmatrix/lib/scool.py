import logging
import os

import cooler
import numpy as np

from .matrixFile import MatrixFile

log = logging.getLogger(__name__)

class Scool(MatrixFile):

    def __init__(self, pMatrixFile=None):
        super().__init__(pMatrixFile)
        log.debug('scool object created')
        self.coolObjectsList = None
        self.bins = None
        self.pixel_list = None
        self.name_list = None

    def load(self):
        raise NotImplementedError('Please use the specific cell to load the individual cool file from the scool file')

    def save(self, pFileName, pSymmetric=True, pApplyCorrection=True):

        pixel_dict = {}
        bins_dict = {}

        if self.coolObjectsList is not None:
            for coolObject in self.coolObjectsList:
                bins_data_frame, matrix_data_frame, dtype_pixel, _ = coolObject.matrixFile.create_cooler_input(pSymmetric=pSymmetric, pApplyCorrection=pApplyCorrection)
                bins_dict[coolObject.matrixFile.matrixFileName] = bins_data_frame
                pixel_dict[coolObject.matrixFile.matrixFileName] = matrix_data_frame

        else:
            try:
                dtype_pixel = {'bin1_id': np.int32, 'bin2_id': np.int32, 'count': self.pixel_list[0]['count'].dtype}
                # dtype_pixel = self.pixel_list[0]['count'].dtype

                for i, pixels in enumerate(self.pixel_list):
                    bins_dict[self.name_list[i]] = self.bins
                    pixel_dict[self.name_list[i]] = pixels
                    log.debug('self.name_list[i] %s', self.name_list[i])
            except Exception as exp:  # pylint: disable=W0718
                log.debug('Exception %s', str(exp))

        local_temp_dir = os.path.dirname(os.path.realpath(pFileName))

        cooler.create_scool(cool_uri=pFileName, bins=bins_dict, cell_name_pixels_dict=pixel_dict,
                            dtypes=dtype_pixel,
                            ordered=True,
                            temp_dir=local_temp_dir)
