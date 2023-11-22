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
         TBB::tbb
         TBB::tbbmalloc)
