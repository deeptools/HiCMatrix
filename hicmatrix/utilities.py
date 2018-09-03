import numpy as np

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