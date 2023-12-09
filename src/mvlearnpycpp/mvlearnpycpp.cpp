#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cluster/mv_coreg_spectral.h"
#include "cluster/mv_spectral.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: mvlearnpycpp

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  py::class_<mvlearn::cluster::MVSpectralClustering>(m, "MVSpectralClustering")
      .def(py::init<int,          // n_clusters
                    int,          // num_samples
                    int,          // num_features,
                    int,          // info_view
                    int,          // max_iter
                    std::string,  // affinity
                    int,          // n_neighbors
                    double,       // gamma
                    bool          // auto_num_clusters
                    >())
      .def("fit", &mvlearn::cluster::MVSpectralClustering::fit)
      .def("fit_predict", &mvlearn::cluster::MVSpectralClustering::fit_predict)
      .def("get_num_clusters",
           &mvlearn::cluster::MVSpectralClustering::get_num_clusters);

  py::class_<mvlearn::cluster::MVCoRegSpectralClustering,
             mvlearn::cluster::MVSpectralClustering>(
      m, "MVCoRegSpectralClustering")
      .def(py::init<int,          // n_clusters
                    int,          // num_samples
                    int,          // num_features,
                    int,          // info_view
                    int,          // max_iter
                    std::string,  // affinity
                    int,          // n_neighbors
                    double,       // gamma
                    bool          // auto_num_clusters
                    >())
      .def("fit", &mvlearn::cluster::MVCoRegSpectralClustering::fit)
      .def("fit_predict",
           &mvlearn::cluster::MVCoRegSpectralClustering::fit_predict)
      .def("get_num_clusters",
           &mvlearn::cluster::MVCoRegSpectralClustering::get_num_clusters);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
