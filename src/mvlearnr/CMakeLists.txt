add_library(mvlearnR mvlearnr.cpp)
target_link_libraries(mvlearnR PUBLIC MVSpectralClustering
                                      MVCoRegSpectralClustering Rcpp)

target_include_directories(mvlearnR PUBLIC ${DIR_INCLUDE})
