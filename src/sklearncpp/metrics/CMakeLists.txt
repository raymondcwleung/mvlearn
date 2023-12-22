add_library(sklearncppMetricsPairwise
            pairwise.cpp "${DIR_INCLUDE_SKLEARNCPP}/metrics/pairwise.h")
target_link_libraries(sklearncppMetricsPairwise
                      PUBLIC Eigen3::Eigen sklearncppNearestNeighbors -llapack
                      -lblas)
target_include_directories(
  sklearncppMetricsPairwise PUBLIC ${DIR_INCLUDE_SKLEARNCPP} ${DIR_INCLUDE}
                                   ${MLPACK_INCLUDE_DIRS})
