add_library(MVSpectralClustering mv_spectral.cpp
                                 "${DIR_INCLUDE}/cluster/mv_spectral.h")

target_link_libraries(MVSpectralClustering PRIVATE Eigen3::Eigen)
target_include_directories(MVSpectralClustering PRIVATE ${DIR_SRC}
                                                        ${DIR_INCLUDE})
