#include <iostream>
#include <cstring>
#include <python3.9/Python.h>
#include <vector>

PyObject* to_list(const std::vector<int> &cpp_vec) {
    PyObject *r = PyList_New(cpp_vec.size());
    if (! r) {
        goto except;
    }
    for (Py_ssize_t i = 0; i < cpp_vec.size(); ++i) {
        PyObject *item = PyLong_FromLong(cpp_vec[i]);
        if (! item || PyErr_Occurred() || PyList_SetItem(r, i, item)) {
            goto except;
        }
    }
    assert(! PyErr_Occurred());
    assert(r);
    goto finally;
except:
    assert(PyErr_Occurred());
    // Clean up list
    if (r) {
        // No PyList_Clear().
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(r); ++i) {
            Py_XDECREF(PyList_GET_ITEM(r, i));
        }
        Py_DECREF(r);
        r = NULL;
    }
finally:
    return r;
}

int main(int argc, char* argv[])
{
    setenv("PYTHONPATH",".",1);

    Py_Initialize();

    std::vector<int> state {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> deriv {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    PyObject* p_state = to_list(state);
    PyObject* p_deriv = to_list(deriv);

    // PySys_SetArgv(argc, argv);
    // PyObject* module_name = PyString_FromString((char*)"example");

    PyObject* module = PyImport_ImportModule("Derivates");
    if (module == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed to import module\n";
        return 1;
    }

    PyObject* dict = PyModule_GetDict(module);
    if (dict == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed to import __dict__\n";
        return 1;
    }

    std::string py_class_name = "ClassicDeriv";
    PyObject* py_class = PyDict_GetItemString(dict, py_class_name.c_str());
    if(py_class == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed import class " << py_class_name << std::endl;
        return 1;
    }

    PyObject* py_arg_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_arg_tuple, 0, PyLong_FromLong(10));
    PyTuple_SetItem(py_arg_tuple, 1, PyLong_FromLong(5));

    PyObject* obj;
    if (PyCallable_Check(py_class))
        obj = PyObject_CallObject(py_class, py_arg_tuple);
    else
        std::cout << "Cannot instantiate the Python class" << std::endl;

    PyObject* val = PyObject_CallMethod(obj, "randomize", NULL);
    if (!val)
        PyErr_Print();

    PyObject* solve_tuple = PyTuple_New(2);
    PyTuple_SetItem(solve_tuple, 0, p_state);
    PyTuple_SetItem(solve_tuple, 1, p_deriv);

    PyObject* solve = PyObject_CallMethod(
      obj, "solve", 
        "([items], [items])", solve_tuple
        // "[i,i,i,i,i,i,i,i,i,i]", 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,   
        // "[i,i,i,i,i,i,i,i,i,i]", 0,  0,  0,  0,  0,  0,  0,  0,  0,  0
      );
    if (!solve)
        PyErr_Print();

    // std::string s (PyUnicode_AsUTF8(val));
    // std::cout << s;

    Py_Finalize();

    return 0;
}