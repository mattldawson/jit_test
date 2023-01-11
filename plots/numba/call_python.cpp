#include <iostream>
#include <cstring>
#include <python3.9/Python.h>

int main(int argc, char* argv[])
{
    setenv("PYTHONPATH",".",1);

    Py_Initialize();

    // PySys_SetArgv(argc, argv);
    // PyObject* module_name = PyString_FromString((char*)"example");

    PyObject* module = PyImport_ImportModule("example");
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

    std::string py_class_name = "point";
    PyObject* py_class = PyDict_GetItemString(dict, py_class_name.c_str());
    if(py_class == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed import class " << py_class_name << std::endl;
        return 1;
    }

    PyObject* py_arg_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_arg_tuple, 0, PyLong_FromLong(5));
    PyTuple_SetItem(py_arg_tuple, 1, PyLong_FromLong(10));

    PyObject* obj;
    if (PyCallable_Check(py_class))
        obj = PyObject_CallObject(py_class, py_arg_tuple);
    else
        std::cout << "Cannot instantiate the Python class" << std::endl;

    PyObject* val = PyObject_CallMethod(obj, "print_position", NULL);
    if (!val)
        PyErr_Print();

    std::string s (PyUnicode_AsUTF8(val));
    std::cout << s;

    Py_Finalize();

    return 0;
}