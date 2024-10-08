cmake_minimum_required(VERSION 3.20..3.28)

# Required libraries / packages
set(CMAKE_TOOLCHAIN_FILE "./vcpkg/scripts/buildsystems/vcpkg.cmake")

set(CONDA_ENV "/PHShome/zl160/miniforge3/envs/leung-multiview-mini")

# Set compilers
#set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
#set(CMAKE_C_COMPILER_WORKS 1)
#set(CMAKE_CXX_COMPILER_WORKS 1)
#set(CMAKE_MAKE_PROGRAM "${CONDA_ENV}/bin/ninja")
#set(CMAKE_CXX_COMPILER "${CONDA_ENV}/bin/x86_64-conda-linux-gnu-g++")
#set(CMAKE_C_COMPILER "${CONDA_ENV}/bin/x86_64-conda-linux-gnu-gcc")
#set(CMAKE_Fortran_COMPILER "${CONDA_ENV}/bin/x86_64-conda-linux-gnu-gfortran")

# C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(ENV{OMP_NUM_THREADS} 4)

# Needed for vcpkg
project(MyPoem VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE Debug)


# IDE prettiness
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(Catch2 3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo REQUIRED)
set(Eigen3_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-linux/share/eigen3)
find_package(Eigen3 CONFIG NO_MODULE REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(Spectra CONFIG REQUIRED)

find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")

add_library(Rcpp INTERFACE)
target_include_directories(Rcpp
                           INTERFACE "${CONDA_ENV}/lib/R/library/Rcpp/include/")

# Use BLAS and LAPACK for Eigen
#add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE_STRICT)


# # For Catch2 include(CTest) include(Catch)

add_definitions(-DARMA_DONT_USE_WRAPPERS)

# Include source directory
set(DIR_SRC "${PROJECT_SOURCE_DIR}/src")
set(DIR_INCLUDE "${PROJECT_SOURCE_DIR}/include")

set(DIR_INCLUDE_SCIPYCPP "${PROJECT_SOURCE_DIR}/include/scipycpp")
set(DIR_SRC_SCIPYCPP "${PROJECT_SOURCE_DIR}/src/scipycpp")

set(DIR_INCLUDE_SKLEARNCPP "${PROJECT_SOURCE_DIR}/include/sklearncpp")
set(DIR_SRC_SKLEARNCPP "${PROJECT_SOURCE_DIR}/src/sklearncpp")

set(DIR_INCLUDE_UTILS_EIGENARMA "${PROJECT_SOURCE_DIR}/include/utils_eigenarma")
set(DIR_SRC_UTILS_EIGENARMA "${PROJECT_SOURCE_DIR}/src/utils_eigenarma")

add_subdirectory(${DIR_SRC})
# add_subdirectory("${CMAKE_SOURCE_DIR}/tests_cpp")
#

install(TARGETS _core DESTINATION mvlearnpycpp)
