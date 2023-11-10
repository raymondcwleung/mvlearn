cmake_minimum_required(VERSION 3.15..3.28)
# Needed for vcpkg
set(CMAKE_TOOLCHAIN_FILE "./vcpkg/scripts/buildsystems/vcpkg.cmake")
project(MyPoem VERSION 0.1.0)

# IDE prettiness
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# mlpack
find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Armadillo
find_package(Armadillo REQUIRED)

# Eigen
set(Eigen3_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-linux/share/eigen3)
find_package(Eigen3 CONFIG NO_MODULE REQUIRED)

# Include source directory
add_subdirectory("${CMAKE_SOURCE_DIR}/src")
