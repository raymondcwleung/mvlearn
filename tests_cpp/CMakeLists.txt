add_executable(testDistance "spatial/distance/test_distance.cpp")
target_link_libraries(testDistance PUBLIC Eigen3::Eigen Catch2::Catch2WithMain)
target_include_directories(testDistance PRIVATE "${CMAKE_SOURCE_DIR}/src/")

add_executable(testPairwise "metrics/pairwise/test_pairwise.cpp")
target_link_libraries(testPairwise PUBLIC Eigen3::Eigen Catch2::Catch2WithMain)
target_include_directories(testPairwise PRIVATE "${CMAKE_SOURCE_DIR}/src/")

catch_discover_tests(testDistance)
catch_discover_tests(testPairwise)
