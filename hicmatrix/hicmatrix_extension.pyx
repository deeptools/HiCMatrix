# cython: language_level=3, boundscheck=False
import numpy as np
cimport numpy as np 

DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t


# np.float16

# np.float16

def fillLowerTriangle(np.ndarray pIndptr, np.ndarray pIndices, np.ndarray pData, DTYPE_INT_t pNnz, DTYPE_INT_t pShape):
    cdef np.ndarray instances = np.zeros(pNnz * 2, dtype=DTYPE_INT)
    cdef np.ndarray features = np.zeros(pNnz * 2, dtype=DTYPE_INT)
    cdef np.ndarray data = np.zeros(pNnz * 2, dtype=np.float16)
    cdef DTYPE_INT_t i = 0
    cdef DTYPE_INT_t counter = 0
    cdef DTYPE_INT_t length = pIndptr.shape[0] - 1
    # log.debug('np arrys init done')
    cdef int j_start = 0
    cdef int j_end = 0
    while i < length:
        j_start = pIndptr[i]
        j_end = pIndptr[i+1]
        
        while j_start < j_end:
            # j = 
            if i == pIndices[j_start]:
                instances[counter] = i
                features[counter] = pIndices[j_start]
                data[counter] = pData[j_start]
                j_start += 1
                counter += 1
            else:
                instances[counter] = i
                features[counter] = pIndices[j_start]
                data[counter] = pData[j_start]
                # j_start += 1
                counter += 1

                instances[counter] = pIndices[j_start]
                features[counter] = i
                data[counter] = pData[j_start]
                counter += 1
                j_start += 1
        i += 1

    cdef np.ndarray index_sort = np.lexsort((features, instances))
    instances = instances[index_sort]
    features = features[index_sort]
    data = data[index_sort]


    cdef np.ndarray indptr = np.zeros(pShape + 1)
    indptr[0] = 0
    counter = 1
    cdef int x = 0
    for x in range(features.shape[0] - 1):
        if features[x] < features[x  + 1]:
            continue
        else:
            indptr[counter] = x
            counter += 1
    indptr[counter] = features.shape[0]
    # x_shape, y_shape = pMatrix.shape
    # del self.matrix
    return indptr, features, data