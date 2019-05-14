#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

static PyObject* fillLowerTriangle(PyObject* self, PyObject* args) {
    
    PyObject *indptrListObj=NULL, *indicesListObj=NULL, *dataListObj=NULL;
    PyObject *indptrList=NULL, *indicesList=NULL, *dataList=NULL;

    if (!PyArg_ParseTuple(args, "OOO", 
                            &indptrListObj, 
                            &indicesListObj,
                            &dataListObj,
                           ))
        return NULL;
    
    indptrList = PyArray_FROM_OTF(indptrListObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    indicesList = PyArray_FROM_OTF(indicesListObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    dataList = PyArray_FROM_OTF(dataListObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    size_t i = 0;
    size_t j_start = 0;
    size_t j_end = 0;
    size_t size_of_indptr =  sizeof(indptrList)/sizeof(indptrList[0]);
    std::cout << "start to parse values" << std::endl;
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


}


// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef nearestNeighborsFunctions[] = {
    {"fillLowerTriangle", fit, METH_VARARGS, "Fills the lower triangle of a matrix"},
   
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef nearestNeighborsModule = {
    PyModuleDef_HEAD_INIT,
    "_hicmatrix",
    NULL,
    -1,
    nearestNeighborsFunctions
};
// definition of the module for python
// PyMODINIT_FUNC
// init_nearestNeighbors(void)
// {
//     (void) Py_InitModule("_nearestNeighbors", nearestNeighborsFunctions);
// }

// definition of the module for python
PyMODINIT_FUNC
PyInit__nearestNeighbors(void)
{
    return PyModule_Create(&nearestNeighborsModule);
    // (void) Py_InitModule("_nearestNeighbors", nearestNeighborsModule);
}


namespace py = pybind11;

py::array_t<double> make_array(const py::ssize_t size) {
    // No pointer is passed, so NumPy will allocate the buffer
    return py::array_t<double>(size);
}

PYBIND11_MODULE(my_module, m) {
    .def("make_array", &make_array,
         py::return_value_policy::move); // Return policy can be left default, i.e. return_value_policy::automatic
}