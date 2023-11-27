add_library(MVSpectralClustering mv_spectral.cpp
                                 "${DIR_INCLUDE}/cluster/mv_spectral.h")

target_include_directories(MVSpectralClustering PUBLIC ${MLPACK_INCLUDE_DIRS}
                                                       ${DIR_INCLUDE})
target_link_libraries(
  MVSpectralClustering
  PUBLIC Eigen3::Eigen
         sklearncppMetricsPairwise
         scipycppSpatialDistance
         sklearncppClusterKMeans
         sklearncppNearestNeighbors
         OpenMP::OpenMP_CXX
         TBB::tbb
         TBB::tbbmalloc)

# target_compile_options(MVSpectralClustering PUBLIC -O3 -Ofast -ffast-math
# -fPIC)
target_compile_options(MVSpectralClustering PUBLIC -Ofast -fPIC)
