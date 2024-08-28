add_library(
  sklearncppNearestNeighbors
  nearestneighbors.cpp "${DIR_INCLUDE_SKLEARNCPP}/neighbors/nearestneighbors.h")
target_link_libraries(sklearncppNearestNeighbors PUBLIC Eigen3::Eigen)
target_include_directories(
  sklearncppNearestNeighbors PRIVATE ${DIR_INCLUDE_SKLEARNCPP} ${DIR_INCLUDE}
                                     ${MLPACK_INCLUDE_DIRS})
