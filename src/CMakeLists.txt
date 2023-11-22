add_subdirectory("cluster")
add_subdirectory("scipycpp")
add_subdirectory("sklearncpp")
add_subdirectory("scikit_build_example")
add_subdirectory("mvlearn")

add_executable(main main.cpp)
target_link_libraries(main MVSpectralClustering)

# MVSpectralClustering) sklearncppNearestNeighbors sklearncppClusterKMeans
# sklearncppMetricsPairwise Eigen3::Eigen)
target_include_directories(main PUBLIC ${DIR_INCLUDE})
