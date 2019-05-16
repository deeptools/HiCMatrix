#include <numpy/arrayobject.h>
#include <Python.h>
#include <iostream>
#include <vector>

static PyObject* fillLowerTriangle(PyObject* self, PyObject* args) {
    
    PyObject *indptrListObj=NULL, *indicesListObj=NULL, *dataListObj=NULL;
    PyObject *indptrList=NULL, *indicesList=NULL, *dataList=NULL;

    if (!PyArg_ParseTuple(args, "OOO", 
                            &indptrListObj, 
                            &indicesListObj,
                            &dataListObj
                           ))
        return NULL;
    
    indptrList = PyArray_FROM_OTF(indptrListObj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    indicesList = PyArray_FROM_OTF(indicesListObj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    dataList = PyArray_FROM_OTF(dataListObj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

    size_t i = 0;
    size_t j_start = 0;
    size_t j_end = 0;
    size_t size_of_indptr =  sizeof(indptrList)/sizeof(indptrList[0]);
    std::cout << "start to parse values" << std::endl;
    // PyObject * instances = PyList_New(sizeOfNeighborList);
    // PyObject * features = PyList_New(sizeOfNeighborList);
    // PyObject * data = PyList_New(sizeOfNeighborList);

    std::vector<size_t> instances;
    std::vector<size_t> features;
    std::vector<size_t> data;

    while (i < size_of_indptr - 1) {
        j_start = indptrList[i];
        j_end = indptrList[i+1];

        while (j_start < j_end) {
          
            if (i == indicesList[j_start]) {
                instances.push_back(i);
                featues.push_back(indicesList[j_start]);
                data.push_back(dataList[j_start]);
            }
            else {
                instances.push_back(i);
                featues.push_back(indicesList[j_start]);
                data.push_back(dataList[j_start]);

                instances.push_back(indicesList[j_start]);
                featues.push_back(i);
                data.push_back(dataList[j_start]);
            }
            j_start++;

        }
        i++;
    }

    std::vector<size_t> indptr;
    size_t counter = 0;
    indptr.push_back(0);
    for (size_t j = 0; j < features.size() - 1; j++) {
        if (features[j] != features[j + 1]) {
            indptr.push_back(j);
        }
    }
    indptr.push_back(features.size());

    PyObject * indptrPythonObj = PyList_New(indptr.size());
    PyObject * indicesPythonObj = PyList_New(features.size());
    PyObject * dataPythonObj = PyList_New(data.size());

    for (size_t i = 0; i < indptr.size(); i++) {
        PyObject* valueNeighbor = Py_BuildValue("i", static_cast<int>(indptr[i]));
        PyList_SetItem(indptrPythonObj, i, valueNeighbor);
    }
    for (size_t i = 0; i < features.size(); i++) {
        PyObject* valueNeighbor = Py_BuildValue("i", static_cast<int>(features[i]));
        PyList_SetItem(indicesPythonObj, i, valueNeighbor);
    }
    for (size_t i = 0; i < data.size(); i++) {
        PyObject* valueNeighbor = Py_BuildValue("i", static_cast<int>(data[i]));
        PyList_SetItem(dataPythonObj, i, valueNeighbor);
    }

    PyObject * returnList;
    returnList = PyList_New(3);
    PyList_SetItem(returnList, 0, indptrPythonObj);
    PyList_SetItem(returnList, 1, indicesPythonObj);
    PyList_SetItem(returnList, 2, dataPythonObj);

    return returnList;
}


// definition of avaible functions for python and which function parsing function in c++ should be called.
static PyMethodDef hicMatrixFunctions[] = {
    {"fillLowerTriangle", fillLowerTriangle, METH_VARARGS, "Fills the lower triangle of a matrix"},
   
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef hicMatrixModule = {
    PyModuleDef_HEAD_INIT,
    "_hicMatrixExtension",
    NULL,
    -1,
    hicMatrixFunctions
};
// definition of the module for python
// PyMODINIT_FUNC
// init_nearestNeighbors(void)
// {
//     (void) Py_InitModule("_nearestNeighbors", nearestNeighborsFunctions);
// }

// definition of the module for python
PyMODINIT_FUNC
PyInit__hicMatrixExtension(void)
{
    return PyModule_Create(&hicMatrixModule);
    // (void) Py_InitModule("_nearestNeighbors", nearestNeighborsModule);
}

