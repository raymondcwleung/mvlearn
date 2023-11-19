add_library(MVSpectralClustering mv_spectral.cpp
                                 "${DIR_INCLUDE}/cluster/mv_spectral.h")

target_link_libraries(MVSpectralClustering PUBLIC Eigen3::Eigen
                                                  sklearncppMetricsPairwise)
target_include_directories(MVSpectralClustering PUBLIC ${DIR_INCLUDE}
                                                       ${MLPACK_INCLUDE_DIRS})
