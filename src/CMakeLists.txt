add_subdirectory("cluster")
add_subdirectory("scipycpp")
add_subdirectory("sklearncpp")
add_subdirectory("mvlearnpycpp")
# add_subdirectory("mvlearnr")

add_executable(main main.cpp)
target_link_libraries(main PUBLIC MVSpectralClustering
                                  MVCoRegSpectralClustering)
target_include_directories(main PUBLIC ${DIR_INCLUDE})
target_compile_options(main PUBLIC -Wall -O3 -llapack -lblas)
