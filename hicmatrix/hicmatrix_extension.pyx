# cython: language_level=3, boundscheck=False
import numpy as np
cimport numpy as np 
from libc.stdio cimport printf
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t


# np.float16

# np.float16

def fillLowerTriangle(np.ndarray pIndptr, np.ndarray pIndices, np.ndarray pData, DTYPE_INT_t pNnz, DTYPE_INT_t pShape):
    cdef np.ndarray instances_init = np.zeros(pNnz, dtype=pIndptr.dtype)
    # cdef np.ndarray features = np.zeros(pNnz, dtype=pIndices.dtype)
    # cdef np.ndarray data = np.zeros(pNnz, dtype=pData.dtype)
    cdef DTYPE_INT_t i = 0
    cdef DTYPE_INT_t counter = 0
    cdef DTYPE_INT_t length = pIndptr.shape[0] - 1
    # log.debug('np arrys init done')
    cdef int j_start = 0
    cdef int j_end = 0
    cdef int nonZeroNonDiagonalCounter = 0
    cdef int diagonalCounter = 0

    while i < length:
        j_start = pIndptr[i]
        j_end = pIndptr[i+1]
        
        while j_start < j_end:
            # j = 
            instances_init[counter] = i
            # features[counter] = pIndices[j_start]
                # data[counter] = pData[j_start]
           
            if i != pIndices[j_start]:
                nonZeroNonDiagonalCounter += 1
            else:
                diagonalCounter += 1
            # else:
            #     instances[counter] = i
            #     features[counter] = pIndices[j_start]
            #     # data[counter] = pData[j_start]
            #     # j_start += 1
            #     counter += 1

            #     # instances[counter] = pIndices[j_start]
            #     # features[counter] = i
            #     # data[counter] = pData[j_start]
            #     # counter += 1
            #     j_start += 1
            j_start += 1
            counter += 1
        i += 1

    cdef np.ndarray instances = np.zeros(nonZeroNonDiagonalCounter*2 + diagonalCounter, dtype=pIndptr.dtype)
    cdef np.ndarray features = np.zeros(nonZeroNonDiagonalCounter*2 + diagonalCounter, dtype=pIndices.dtype)
    cdef np.ndarray data = np.zeros(nonZeroNonDiagonalCounter*2 + diagonalCounter, dtype=pData.dtype)

    printf('%i\n', nonZeroNonDiagonalCounter*2 + diagonalCounter)
    printf('%i\n', nonZeroNonDiagonalCounter)

    printf('%i\n', diagonalCounter)

    i = 0
    counter = 0
    while i < instances_init.shape[0] - 1:
        if instances_init[i] == pIndices[i]:
            instances[counter] = instances_init[i]
            features[counter] = pIndices[i]
            data[counter] = pData[i]
            counter += 1
        else:
            instances[counter] = instances_init[i]
            features[counter] = pIndices[i]
            data[counter] = pData[i]
            counter += 1

            instances[counter] = pIndices[i]
            features[counter] = instances_init[i]
            data[counter] = pData[i]
            counter += 1
        i += 1

    cdef np.ndarray index_sort = np.lexsort((features, instances))
    instances = instances[index_sort]
    features = features[index_sort]
    data = data[index_sort]


    cdef np.ndarray indptr = np.zeros(pShape + 1)
    indptr[0] = 0
    counter = 1
    # cdef int counter_2 = 0
    # cdef int counter_1 = 0
    # cdef int tmp0 = 0
    # cdef int tmp1 = 0
    # cdef int tmp2 = 0


    i = 0
    while i < (instances.shape[0] - 1):
        # tmp0 = instances[i]
        # tmp1 = instances[i+1]
        # tmp2 = instances[i+1] +1 

        # printf("instances[i] %i\n", tmp0)
        # printf("instances[i+1] %i\n", tmp1)
        # printf("instances[i+1]+1 %i\n", tmp2)
        # printf("\n")



        if instances[i] == instances[i  + 1]:
            i += 1
            continue
        elif instances[i] + 1 == instances[i  + 1]:
            indptr[counter] = i
            counter += 1
            # counter_1 += 1

        else:
            indptr[counter] = indptr[counter - 1]
            counter += 1
            # counter_2 += 1
        i += 1
        # printf("%i\n", i)
    indptr[counter] = features.shape[0]
    # x_shape, y_shape = pMatrix.shape
    # del self.matrix
    return indptr, features, data
    # return instances, pIndices