#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
/**/
#include <Eigen/Dense>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void sum_twodim(py::EigenDRef<Eigen::MatrixXd> m) {
  constexpr double my_const{4.0};

  for (int i = 0; i < m.rows(); i++)
    for (int j = 0; j < m.cols(); j++) m(i, j) = my_const * m(i, j);
}

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(mvlearncpp, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc",
        py::arg("i"), py::arg("j"));

  /* m.def( */
  /*     "subtract", [](int i, int j) { return i - j; }, R"pbdoc( */
  /*       Subtract two numbers */
  /**/
  /*       Some other explanation about the subtract function. */
  /*   )pbdoc"); */
  /**/
  /* m.def("mult_three", &mult_three, R"pbdoc(Multiply by three)pbdoc"); */

  m.def("sum_twodim", &sum_twodim, R"pbdoc(Add 2d matrix)pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

int main() { return 0; };
