add_executable(main main.cpp)
target_include_directories(main PRIVATE ${MLPACK_INCLUDE_DIRS})

target_link_libraries(main PUBLIC MVSpectralClustering)
target_include_directories(main PUBLIC ${PROJECT_INCLUDE_DIR})

# include_directories(${PYTHON_INCLUDE_DIRS}) pybind11_add_module(mvlearncpp
# mvlearn.cpp) target_link_libraries(mvlearncpp PUBLIC ${PYTHON_LIBRARIES}
# Eigen3::Eigen)

# add_library(UtilsArmaEigen utilsarmaeigen.cpp
# ${PROJECT_SOURCE_DIR}/include/utilsarmaeigen.h)
# target_link_libraries(UtilsArmaEigen PUBLIC Eigen3::Eigen Armadillo)
# target_include_directories(UtilsArmaEigen PRIVATE
# ${PROJECT_SOURCE_DIR}/include)

add_subdirectory("cluster")
add_subdirectory("utils_eigenarma")
add_subdirectory("scipycpp")
add_subdirectory("sklearncpp")
