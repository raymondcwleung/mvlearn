python_add_library(_core MODULE scikit_build_main.cpp WITH_SOABI)
target_link_libraries(_core PUBLIC pybind11::headers MVSpectralClustering)
target_include_directories(_core PUBLIC ${DIR_INCLUDE})
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
