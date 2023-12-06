import logging

from scipy.sparse import triu

from .matrixFile import MatrixFile

log = logging.getLogger(__name__)

class Ginteractions(MatrixFile):

    def __init__(self, pMatrixFile):
        super().__init__(pMatrixFile)

    def load(self):
        log.error('Not implemented')

    def save(self, pFileName, pSymmetric=None, pApplyCorrection=None):

        # self.restoreMaskedBins()
        log.debug(self.matrix.shape)
        mat_coo = triu(self.matrix, k=0, format='csr').tocoo()
        with open(f"{pFileName}.tsv", 'w', encoding='utf-8') as fileh:
            for idx, counts in enumerate(mat_coo.data):
                chr_row, start_row, end_row, _ = self.cut_intervals[mat_coo.row[idx]]
                chr_col, start_col, end_col, _ = self.cut_intervals[mat_coo.col[idx]]
                fileh.write(f"{chr_row}\t{int(start_row)}\t{int(end_row)}\t{chr_col}\t{int(start_col)}\t{int(end_col)}\t{counts}\n")
