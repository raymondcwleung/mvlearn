pybind11_add_module(mvLearn mvlearn.cpp)
target_link_libraries(mvLearn PUBLIC Eigen3::Eigen clusterRotate)
