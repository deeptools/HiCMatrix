import importlib
import logging

log = logging.getLogger(__name__)


class MatrixFileHandler():
    """
    This class handles the load and save of the different Hi-C contact matrix formats.
    """

    def __init__(self, pFileType='cool', pMatrixFile=None, pChrnameList=None,
                 pApplyCorrectionCoolerLoad=None, pBedFileHicPro=None, pCorrectionFactorTable=None,
                 pCorrectionOperator=None, pEnforceInteger=None, pAppend=None, pFileWasH5=None, pHiCInfo=None, pHic2CoolVersion=None,
                 pDistance=None, pMatrixFormat=None, pLoadMatrixOnly=None, pNoCutIntervals=None):

        self.class_ = getattr(importlib.import_module('.' + pFileType.lower(), package='hicmatrix.lib'), pFileType.title())

        if pFileType == 'hicpro':
            self.matrixFile = self.class_(pMatrixFile=pMatrixFile, pBedFile=pBedFileHicPro)
        else:
            self.matrixFile = self.class_(pMatrixFile=pMatrixFile)
            if pFileType == 'cool':
                self.matrixFile.chrnameList = pChrnameList
                if pCorrectionFactorTable is not None:
                    self.matrixFile.correctionFactorTable = pCorrectionFactorTable
                if pCorrectionOperator is not None:
                    self.matrixFile.correctionOperator = pCorrectionOperator
                if pEnforceInteger is not None:
                    self.matrixFile.enforceInteger = pEnforceInteger
                if pAppend is not None:
                    self.matrixFile.appendData = pAppend
                if pFileWasH5 is not None:
                    self.matrixFile.fileWasH5 = pFileWasH5
                if pApplyCorrectionCoolerLoad is not None:
                    self.matrixFile.applyCorrectionLoad = pApplyCorrectionCoolerLoad
                if pHiCInfo is not None:
                    self.matrixFile.hic_metadata = pHiCInfo
                if pHic2CoolVersion is not None:
                    self.matrixFile.hic2cool_version = pHic2CoolVersion
                if pDistance is not None:
                    self.matrixFile.distance = pDistance
                if pMatrixFormat is not None:
                    self.matrixFile.matrixFormat = pMatrixFormat
                if pLoadMatrixOnly is not None:
                    self.matrixFile.matrixOnly = pLoadMatrixOnly
                if pNoCutIntervals is not None:
                    self.matrixFile.noCutIntervals = pNoCutIntervals

    def load(self):

        return self.matrixFile.load()

    def set_matrix_variables(self, pMatrix, pCutIntervals, pNanBins, pCorrectionFactors, pDistanceCounts):
        self.matrixFile.set_matrix_variables(pMatrix, pCutIntervals, pNanBins, pCorrectionFactors, pDistanceCounts)

    def save(self, pName, pSymmetric, pApplyCorrection):
        self.matrixFile.save(pName, pSymmetric, pApplyCorrection)

    def load_init(self):
        pass
