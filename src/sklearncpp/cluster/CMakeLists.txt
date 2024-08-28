add_library(sklearncppClusterKMeans
            kmeans.cpp "${DIR_INCLUDE_SKLEARNCPP}/cluster/kmeans.h")
target_link_libraries(sklearncppClusterKMeans PUBLIC Eigen3::Eigen
    ${ARMADILLO_LIBRARIES})
target_include_directories(
  sklearncppClusterKMeans PUBLIC ${MLPACK_INCLUDE_DIRS} ${DIR_INCLUDE}
  ${DIR_INCLUDE_SKLEARNCPP} ${ARMADILLO_INCLUDE_DIRS})
