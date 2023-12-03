add_library(evrot Evrot.cpp "${DIR_INCLUDE}/cluster/Evrot.h")
target_link_libraries(evrot PUBLIC Eigen3::Eigen)
target_include_directories(evrot PUBLIC ${DIR_INCLUDE})

add_library(clusterRotate ClusterRotate.cpp
                          "${DIR_INCLUDE}/cluster/ClusterRotate.h")
target_link_libraries(clusterRotate PUBLIC Eigen3::Eigen evrot)
target_include_directories(clusterRotate PUBLIC ${DIR_INCLUDE})

add_library(MVSpectralClustering mv_spectral.cpp
                                 "${DIR_INCLUDE}/cluster/mv_spectral.h")
target_include_directories(MVSpectralClustering PUBLIC ${MLPACK_INCLUDE_DIRS})
target_link_libraries(
  MVSpectralClustering
  PUBLIC Eigen3::Eigen
         sklearncppMetricsPairwise
         scipycppSpatialDistance
         sklearncppClusterKMeans
         sklearncppNearestNeighbors
         clusterRotate
         OpenMP::OpenMP_CXX
         TBB::tbb
         TBB::tbbmalloc)

# target_compile_options(MVSpectralClustering PUBLIC -O3 -Ofast -ffast-math
# -fPIC) target_compile_options(MVSpectralClustering PUBLIC -Ofast -fPIC)

add_library(MVCoRegSpectralClustering
            mv_coreg_spectral.cpp "${DIR_INCLUDE}/cluster/mv_coreg_spectral.h")
target_include_directories(MVCoRegSpectralClustering
                           PUBLIC ${MLPACK_INCLUDE_DIRS})
target_link_libraries(
  MVCoRegSpectralClustering PUBLIC MVSpectralClustering sklearncppClusterKMeans
                                   Eigen3::Eigen)
