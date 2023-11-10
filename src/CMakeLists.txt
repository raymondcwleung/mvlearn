include_directories(${PYTHON_INCLUDE_DIRS})
pybind11_add_module(mvlearncpp mvlearn.cpp)
target_link_libraries(mvlearncpp PUBLIC ${PYTHON_LIBRARIES} Eigen3::Eigen)

add_executable(main main.cpp)
target_include_directories(main PRIVATE ${MLPACK_INCLUDE_DIRS})

add_library(UtilsArmaEigen utilsarmaeigen.cpp
                           ${PROJECT_SOURCE_DIR}/include/utilsarmaeigen.h)
target_link_libraries(UtilsArmaEigen PUBLIC Eigen3::Eigen Armadillo)
target_include_directories(UtilsArmaEigen PRIVATE ${PROJECT_SOURCE_DIR}/include)
