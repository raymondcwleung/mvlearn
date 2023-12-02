#include <cluster/mv_spectral.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  py::class_<mvlearn::cluster::MVSpectralClustering>(m, "MVSpectralClustering")
      .def(py::init<int,          // n_clusters
                    int,          // num_samples
                    int,          // num_features,
                    int,          // random_state
                    int,          // info_view
                    int,          // max_iter
                    int,          // n_init,
                    std::string,  // affinity
                    int,          // n_neighbors
                    double,       // gamma
                    bool          // auto_num_clusters
                    >())
      .def("fit", &mvlearn::cluster::MVSpectralClustering::fit)
      .def("fit_predict", &mvlearn::cluster::MVSpectralClustering::fit_predict)
      .def("get_num_clusters",
           &mvlearn::cluster::MVSpectralClustering::get_num_clusters);

  m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
