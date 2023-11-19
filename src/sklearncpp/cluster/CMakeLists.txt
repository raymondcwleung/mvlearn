add_library(sklearncppClusterKMeans
            kmeans.cpp "${DIR_INCLUDE_SKLEARNCPP}/cluster/kmeans.h")
target_link_libraries(sklearncppClusterKMeans PUBLIC Eigen3::Eigen)
target_include_directories(
  sklearncppClusterKMeans PUBLIC ${DIR_INCLUDE} ${DIR_INCLUDE_SKLEARNCPP}
                                 ${MLPACK_INCLUDE_DIRS})
