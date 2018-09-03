import numpy as np
import sys


def toString(s):
    """
    This takes care of python2/3 differences
    """
    if isinstance(s, str):
        return s
    if isinstance(s, bytes):  # or isinstance(s, np.bytes_):
        if sys.version_info[0] == 2:
            return str(s)
        return s.decode('ascii')
    if isinstance(s, list):
        return [toString(x) for x in s]
    if isinstance(s, np.ndarray):
        return s.astype(str)
    return s


def toBytes(s):
    """
    Like toString, but for functions requiring bytes in python3
    """
    if sys.version_info[0] == 2:
        return s
    if isinstance(s, bytes):
        return s
    # if isinstance(s, np.bytes_):
    #     return np.bytes_(s)
    if isinstance(s, str):
        return bytes(s, 'ascii')
    if isinstance(s, list):
        return [toBytes(x) for x in s]
    return s


def check_chrom_str_bytes(pIteratableObj, pObj):
    # determine type
    if isinstance(pObj, list) and len(pObj) > 0:
        type_ = type(pObj[0])
    else:
        type_ = type(pObj)
    if not isinstance(type(next(iter(pIteratableObj))), type_):
        if type(next(iter(pIteratableObj))) is str:
            pObj = toString(pObj)
        elif type(next(iter(pIteratableObj))) in [bytes, np.bytes_]:
            pObj = toBytes(pObj)
    return pObj


def convertNansToZeros(ma):
    nan_elements = np.flatnonzero(np.isnan(ma.data))
    if len(nan_elements) > 0:
        ma.data[nan_elements] = 0
    return ma


def convertNansToOnes(pArray):
    nan_elements = np.flatnonzero(np.isnan(pArray))
    if len(nan_elements) > 0:
        pArray[nan_elements] = 1.0
    return pArray


def enlarge_bins(bin_intervals):
    r"""
    takes a list of consecutive, but not
    directly touching, bin intervals
    and joins them such that the
    end and start of consecutive bins
    is the same.

    >>> bin_intervals = [('chr1', 10, 50, 1), ('chr1', 50, 80, 2),
    ... ('chr2', 10, 60, 3), ('chr2', 70, 90, 4)]
    >>> enlarge_bins(bin_intervals)
    [('chr1', 0, 50, 1), ('chr1', 50, 80, 2), ('chr2', 0, 65, 3), ('chr2', 65, 90, 4)]
    """
    # enlarge remaining bins
    chr_start = True
    for idx in range(len(bin_intervals) - 1):
        chrom, start, end, extra = bin_intervals[idx]
        chrom_next, start_next, end_next, extra_next = bin_intervals[idx + 1]

        if chr_start is True:
            start = 0
            chr_start = False
            bin_intervals[idx] = (chrom, start, end, extra)
        if chrom == chrom_next and end != start_next:
            middle = start_next - int((start_next - end) / 2)
            bin_intervals[idx] = (chrom, start, middle, extra)
            bin_intervals[idx + 1] = (chrom, middle, end_next, extra_next)
        if chrom != chrom_next:
            chr_start = True

    chrom, start, end, extra = bin_intervals[-1]
    bin_intervals[-1] = (chrom, start, end, extra)

    return bin_intervals
