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
         Spectra::Spectra
         sklearncppMetricsPairwise
         scipycppSpatialDistance
         sklearncppClusterKMeans
         sklearncppNearestNeighbors
         clusterRotate
         TBB::tbb
         TBB::tbbmalloc
         OpenMP::OpenMP_CXX
         -lopenblas
         -llapacke)
# OpenMP::OpenMP_CXX TBB::tbb TBB::tbbmalloc)
# target_compile_options(MVSpectralClustering PUBLIC -O3 -fopenmp)
target_compile_options(
  MVSpectralClustering
  PUBLIC -O3
         -ffast-math
         -Wall
         -Wextra
         -DEIGEN_NO_DEBUG
         -DNDEBUG
         -fopenmp
         -funroll-loops)

add_library(MVCoRegSpectralClustering
            mv_coreg_spectral.cpp "${DIR_INCLUDE}/cluster/mv_coreg_spectral.h")
target_include_directories(MVCoRegSpectralClustering
                           PUBLIC ${MLPACK_INCLUDE_DIRS})
target_link_libraries(
  MVCoRegSpectralClustering PUBLIC MVSpectralClustering sklearncppClusterKMeans
                                   Eigen3::Eigen)
# OpenMP::OpenMP_CXX TBB::tbb TBB::tbbmalloc)
# target_compile_options(MVCoRegSpectralClustering PUBLIC -O3 -fPIC -fopenmp)
target_compile_options(MVCoRegSpectralClustering PUBLIC -O0 -g)
